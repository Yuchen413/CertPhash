import torch
import os
# os.environ['AUTOLIRPA_DEBUG'] = '1'

from auto_LiRPA import BoundedModule, BoundedTensor, PerturbationLpNorm
import torch.nn as nn
from tqdm import tqdm
from torch.utils.data import DataLoader
from argparse import ArgumentParser
from models.resnet import resnet
from models.resnet_v5 import resnet_v5
from datasets import *
import matplotlib.pyplot as plt

seed_value = 2024
torch.manual_seed(seed_value)
torch.cuda.manual_seed(seed_value)
torch.cuda.manual_seed_all(seed_value)
np.random.seed(seed_value)
random.seed(seed_value)


def get_opts():
    parser = ArgumentParser()
    parser.add_argument(
        "--data-dir", help="Directory containing train/validation images", type=str, default="data"
    )
    parser.add_argument(
        "--train-data",
        help="Training data, CSV of pairs of (path, base64-encoded hash)",
        type=str,
        # required=True,
        default='coco-train.csv'
    )
    parser.add_argument("--val-data", help="Validation data", default='coco-val.csv', type=str, required=False)
    parser.add_argument("--epochs", help="Training epochs", type=int, default=50)
    parser.add_argument("--checkpoint_iter", help="Checkpoint frequency", type=int, default=-1)
    parser.add_argument("--batch_size", help="Batch size", type=int, default=1)
    parser.add_argument("--verbose", help="Print intermediate statistics", action="store_true")
    parser.add_argument("--in_dim", default=64, type=int)
    parser.add_argument('--cut_epsilon',
                        default=8. / 255, type=float)
    parser.add_argument('--verify_epsilon',
                        default=8. / 255, type=float)
    parser.add_argument('--output_threshold',
                        default=90, type=int)
    parser.add_argument('--model',
                        default='../train_verify/mnist_pdq_ep1/ckpt_best.pth', type=str)
    return parser.parse_args()


class SummationLayer(nn.Module):
    def __init__(self, input_features=256):
        super(SummationLayer, self).__init__()
        # Initialize a linear layer with input_features -> 1 output
        self.linear = nn.Linear(input_features, 1, bias=False)

        # Initialize weights to all ones, so that it effectively sums the inputs
        with torch.no_grad():
            self.linear.weight.fill_(1.0)

    def forward(self, x):
        # Apply the linear layer, which sums the inputs
        return self.linear(x)


class ModifiedModel(nn.Module):
    def __init__(self, orig_model,  in_dim=32, channel=3):
        super(ModifiedModel, self).__init__()

        self.orig_model = orig_model
        self.sum_layer = SummationLayer()
        self.in_dim = in_dim
        self.channel = channel

    def forward(self, x):
        x = self.orig_model(x)
        x = torch.relu(x)
        x = self.sum_layer(x)
        return x


def get_preimage(model_ori, val_dataloader, cut_eps, dummy_input, threshold=1):
    preimage_save_path = 'saved_preimages'
    apply_output_constraints_to = ['BoundMatMul', 'BoundInput']
    model = BoundedModule(model_ori, dummy_input, bound_opts={
        "conv_mode": "patches",
        'optimize_bound_args': {
            'apply_output_constraints_to': apply_output_constraints_to,
            'tighten_input_bounds': True,
            'best_of_oc_and_no_oc': True,
            'directly_optimize': ['/input.1'],
            'oc_lr': 0.1,
            'share_gammas': False,
            'iteration': 1000,
        }
    })

    x_list = []
    x_L_list = []
    x_U_list = []
    cls_list =[]
    for i, data in enumerate(tqdm(val_dataloader)):
        if i >= 1000:
            break
        try:
            x, y, cls = data
            x = x.cuda()
            model.eval()
            y_clean = model(x)
            norm = float("inf")
            lower = torch.clamp(x - cut_eps, min=0)
            upper = torch.clamp(x + cut_eps, max=1)
            ptb = PerturbationLpNorm(norm=norm, x_L=lower, x_U=upper, eps=cut_eps)
            bounded_x = BoundedTensor(x, ptb)

            # We need, |y-y_clean|<=90, that is {y-y_clean <=90 and y_clean-y <=90}
            # Two constraints as per inequalities for Hy+d<=0, That is y-(y_clean+90)<=0 and -y+(y_clean-90)<=0
            # So I have:
            model.constraints = torch.tensor([[[1], [-1]]], dtype=torch.float32)  # this is H
            #the correct one should be considering "d = -model.thresholds" in output_constraints.py:
            model.thresholds = torch.tensor([(y_clean + threshold), -(y_clean - threshold)],
                                            dtype=torch.float32)  # this is -d
            model.compute_bounds(x=(bounded_x,), method='CROWN-Optimized')

            tightened_ptb = model['/input.1'].perturbation
            x_L = tightened_ptb.x_L
            x_U = tightened_ptb.x_U


            if float('inf') in x_L or float('inf') in x_U:
                print('Cannot reach the bounds, infinity')
            else:
                x_L = torch.clamp(x_L, min=0)
                x_U = torch.clamp(x_U, max=1)
                x_L_list.append(x_L.squeeze(0).detach().cpu())
                x_U_list.append(x_U.squeeze(0).detach().cpu())
                x_list.append(x.squeeze(0).detach().cpu())
                cls_list.append(cls)
                ori_image = x.squeeze().cpu().detach().numpy()
                lb_image = x_L.squeeze().cpu().detach().numpy()
                ub_image = x_U.squeeze().cpu().detach().numpy()
                combined_image = np.concatenate((ori_image, lb_image, ub_image), axis=1)
                img_save_path = f'{preimage_save_path}/images/dy_{threshold}'
                os.makedirs(img_save_path, exist_ok=True)
                plt.imsave(os.path.join(img_save_path, f'{i}.png'), combined_image, cmap='gray')  # cmap='gray' for grayscale


        except Exception as e:
            if 'CUDA out of memory' in str(e):
                print(f"Error: CUDA out of memory due to the bounds are too loose")
                torch.cuda.empty_cache()
            continue

    if len(x_L_list) > 0:
        os.makedirs(preimage_save_path, exist_ok=True)
        torch.save(torch.stack(x_list), os.path.join(preimage_save_path, f'X_dy_{threshold}_{cut_eps:.4f}.pt'))
        torch.save(torch.stack(x_L_list), os.path.join(preimage_save_path, f'XL_dy_{threshold}_{cut_eps:.4f}.pt'))
        torch.save(torch.stack(x_U_list), os.path.join(preimage_save_path, f'XU_dy_{threshold}_{cut_eps:.4f}.pt'))
        torch.save(torch.tensor(cls_list), os.path.join(preimage_save_path, f'Classes_dy_{threshold}_{cut_eps:.4f}.pt'))  # Save classes
        print(f'Preimages saved at {preimage_save_path}')
        return True
    else:
        print(f'No reachable preimages, certified no collision rate: 0%')
        return False

def verify_preimage(preimage_save_path = 'saved_preimages', threshold = 1, verified_epsilon = 8/255, cut_eps = 64/255):
    X = torch.load(os.path.join(preimage_save_path, f'X_dy_{threshold}_{cut_eps:.4f}.pt')).cuda()
    X_L = torch.load(os.path.join(preimage_save_path, f'XL_dy_{threshold}_{cut_eps:.4f}.pt')).cuda()
    X_U = torch.load(os.path.join(preimage_save_path, f'XU_dy_{threshold}_{cut_eps:.4f}.pt')).cuda()
    classes = torch.load(os.path.join(preimage_save_path, f'Classes_dy_{threshold}_{cut_eps:.4f}.pt'))

    num_sample = len(X)

    #first count the cannot cannot rach bounds
    cannot_reach_preimage = num_sample - len(X_L)
    print(f'Cannot reach: {cannot_reach_preimage}')

    overlap_indices = set()  # Use a set to store indices of overlapping pairs for uniqueness
    for i in range(len(X_L) - 1):
        for j in range(i + 1, len(X_L)):
            if classes[i] != classes[j]:
                overlaps = (X_L[i] <= X_U[j]).all() and (X_U[i] >= X_L[j]).all()
                if overlaps:
                    overlap_indices.update([i, j])  # Add both indices to the set if they overlap
    count_collision_under_cut_eps = len(overlap_indices)  # The number of unique indices involved in overlaps
    print(f'Count preimage overlap under cut eps {cut_eps:.4f}: {count_collision_under_cut_eps}')
    print(f"Verified NO Collision Rate: {(num_sample - cannot_reach_preimage - count_collision_under_cut_eps) / num_sample}")

def main():
    opts = get_opts()
    opts.in_dim = 28
    channel = 1
    opts.val_data = 'data/mnist/mnist_test.csv'
    opts.model = 'saved_models/mnist_pdq_ep8/ckpt_best.pth'

    val_data = ImageToHashAugmented_PDQ_with_class(opts.val_data, opts.data_dir, resize=opts.in_dim, num_augmented=0)
    val_dataloader = DataLoader(val_data, batch_size=opts.batch_size, shuffle=True, pin_memory=True)
    model = resnet(in_ch=1, in_dim=opts.in_dim)
    model_weights = torch.load(opts.model)
    model.load_state_dict(model_weights)

    modified_model = ModifiedModel(model, in_dim=opts.in_dim, channel=channel)
    modified_model.cuda()
    modified_model.eval()


    opts.cut_epsilon = 96/ 255
    opts.output_threshold = 90
    opts.verify_epsilon = 96/ 255


    preimages = get_preimage(modified_model, val_dataloader, cut_eps=opts.cut_epsilon,
                 dummy_input=torch.zeros(1, channel, opts.in_dim, opts.in_dim), threshold=opts.output_threshold)

    if preimages == True:
        verify_preimage(threshold=opts.output_threshold, verified_epsilon=opts.verify_epsilon, cut_eps=opts.cut_epsilon)




if __name__ == "__main__":
    main()
