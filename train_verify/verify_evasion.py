from auto_LiRPA import BoundedModule, BoundedTensor, PerturbationLpNorm
from auto_LiRPA.perturbations import PerturbationL0Norm
from auto_LiRPA.perturbations import PerturbationLpNorm
from datasets import *
from torch.utils.data import DataLoader
from argparse import ArgumentParser
import numpy as np
import base64
import torch.nn as nn
from models.resnet import resnet
from models.resnet_v5 import resnet_v5
from tqdm import tqdm
import argparse
import logging

seed_value = 2024
torch.manual_seed(seed_value)
torch.cuda.manual_seed(seed_value)
torch.cuda.manual_seed_all(seed_value)  # for all GPUs
np.random.seed(seed_value)
random.seed(seed_value)

def post_photodna(x):
    return torch.relu(torch.round(x))

def post_pdq(x):
    labels = torch.relu(x) > 0.5
    return labels.int()

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
    parser.add_argument("--val-data", help="Validation data", default='coco-val.csv',type=str, required=False)
    parser.add_argument("--epochs", help="Training epochs", type=int, default=50)
    parser.add_argument("--checkpoint_iter", help="Checkpoint frequency", type=int, default=-1)
    parser.add_argument("--batch_size", help="Batch size must be 1 to avoid cuda OOM", type=int, default=1, choices=[1])
    parser.add_argument("--verbose", help="Print intermediate statistics", action="store_true")
    parser.add_argument("--in_dim", default=64, type=int)
    parser.add_argument('--epsilon',
                        default=1. / 255, type=float)
    parser.add_argument('--model',
                        default='../train_verify/64_ep1_resv5_l1_aug2_new_collision/ckpt_best.pth', type=str)
    parser.add_argument("--data", type = str,  default='mnist', choices=['coco', 'mnist', 'nsfw'])

    return parser.parse_args()

def tensor_to_hash(y):
    """Convert the model's ouput to a Base64 encoded string (hash)."""
    y = torch.relu(torch.round(y))
    uint8_array = y.cpu().detach().numpy().astype(np.uint8) ##hash bin with int
    byte_array = uint8_array.tobytes()
    encoded_str = base64.b64encode(byte_array).decode("utf-8") ##hashes
    binary_array = np.unpackbits(np.frombuffer(byte_array, dtype=np.uint8))
    binary_strings = ''.join(str(i) for i in binary_array) ##010101010

    return y, binary_strings, encoded_str

def calculate_acc(y_pred = None, y_true=None, hashes_csv = '/home/yuchen/code/verified_phash/Normal-Training/coco-val.csv'):
    hashes = []
    with open(hashes_csv) as f:
        r = csv.reader(f)
        for line in r:
            h = torch.tensor(np.array(list(base64.b64decode(line[1])), dtype=np.uint8)).float()
            hashes.append(h)
    hashes_tensor = torch.stack(hashes).cuda()
    y_pred_expanded = y_pred.unsqueeze(1).expand(-1, hashes_tensor.size(0), -1)
    mse = torch.nn.functional.mse_loss(y_pred_expanded, hashes_tensor.unsqueeze(0), reduction='none').mean(2)
    min_mse_indices = torch.argmin(mse, dim=1)  # Shape [256]
    selected_hashes = hashes_tensor[min_mse_indices]
    correct_matches = torch.all(selected_hashes == y_true, dim=1)
    acc = torch.sum(correct_matches).item()/len(y_true)
    return acc

def get_evasion(model, val_dataloader, eps, dummy_input, threshold=90):
    model.eval()
    evasion = 0
    num_samples = 0
    l = nn.L1Loss(reduction='none')
    model = BoundedModule(model, dummy_input, bound_opts={"conv_mode": "patches"})
    model.eval()

    with torch.no_grad():
        for i, data in enumerate(tqdm(val_dataloader)):
            try:
                x, _= data
                x = x.cuda()
                norm = np.inf
                # norm = 1
                ptb = PerturbationL0Norm(eps=eps)
                bounded_image = BoundedTensor(x, ptb)
                y_clean = post_pdq(model(x))
                lb, ub = model.compute_bounds(x=(bounded_image,), method='CROWN')
                post_lb = post_pdq(lb)
                post_ub = post_pdq(ub)
                loss_eva_lb = l(post_lb, y_clean)
                loss_eva_ub = l(post_ub, y_clean)

                # Evasion
                evasion += torch.sum((max(loss_eva_lb.sum(1), loss_eva_ub.sum(1)) >= threshold)).item()
                # robust_eva_lb = torch.sum(loss_eva_lb.sum(1) >= threshold).item()
                # robust_eva_ub = torch.sum(loss_eva_ub.sum(1) >= threshold).item()
                # evasion += max(robust_eva_ub, robust_eva_lb)

                num_samples += x.size(0)

            except RuntimeError as e:
                if 'CUDA out of memory' in str(e):
                    print(f"Error: CUDA out of memory.")
                    break  # Break out of the loop on CUDA memory error
                else:
                    raise  # Re-raise the exception if it's not a memory error

    # Calculate average loss and accuracy
    # average_loss = total_loss / num_samples
    evasion_rate = evasion / num_samples
    # collision_rate = collision / num_samples

    # print(f"Average L1 Loss: {average_loss:.4f}")
    print(f"Evasion Rate: {evasion_rate * 100:.2f}%")
    # print(f"Collision Rate: {collision_rate * 100:.2f}%")
    return evasion_rate

def main():
    opts = get_opts()

    if opts.data == 'mnist':
        opts.val_data = 'data/mnist/mnist_test.csv'
        resize = 28 ##for resize
        model = resnet(in_ch=1, in_dim=resize)  # PDQ
        dummy_input = torch.zeros(1, 1, 28, 28)
        val_data = ImageToHashAugmented_PDQ(opts.val_data, opts.data_dir, resize, 0) if opts.val_data else None
        val_dataloader = (
            DataLoader(val_data, batch_size=opts.batch_size, shuffle=True, pin_memory=True)
            if val_data
            else None
        )

    else:
        opts.val_data = 'data/coco-val.csv'
        # opts.val_data = 'data/nsfw_val.csv'
        resize = 64 ##for resize
        dummy_input = torch.zeros(1, 3, 64, 64)
        model = resnet_v5(num_classes=144, bn=True, input_dim=resize)  # PhotoDNA
        val_data = ImageToHashAugmented(opts.val_data, opts.data_dir, resize, 0) if opts.val_data else None
        val_dataloader = (
            DataLoader(val_data, batch_size=opts.batch_size, shuffle=True, pin_memory=True)
            if val_data
            else None
        )


    model_weights = torch.load(opts.model)
    model.load_state_dict(model_weights)
    model.cuda()
    get_evasion(model, val_dataloader, eps=opts.epsilon, dummy_input=dummy_input)



if __name__ == "__main__":
    main()
