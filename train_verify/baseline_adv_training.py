import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from argparse import ArgumentParser
from datasets import *
from utils import *
from models.resnet import resnet
from models.resnet_v5 import resnet_v5
import torch.nn.functional as F
from models.wide_resnet_bn_imagenet64 import wide_resnet_bn_imagenet64
from models.resnext_bn_imagenet64 import ResNeXt_bn_imagenet64
# from models.NeuralHash import NeuralHash



seed_value = 2024
torch.manual_seed(seed_value)
torch.cuda.manual_seed(seed_value)
torch.cuda.manual_seed_all(seed_value)  # for all GPUs
np.random.seed(seed_value)
random.seed(seed_value)


def deranged_permutation(n):
    """Generate a derangement of indices for tensor of length n."""
    while True:
        p = torch.randperm(n)
        if all(p != torch.arange(n)):
            return p

def pgd_linf(model, x, y, loss_fct, epsilon=8/255, alpha=5e-4, num_iter=10, randomize=False):
    """ Construct  adversarial examples on the examples X"""
    if randomize:
        delta = torch.rand_like(x, requires_grad=True)
        delta.data = delta.data * 2 * epsilon - epsilon
    else:
        delta = torch.zeros_like(x, requires_grad=True)
    for t in range(num_iter):
        y_p = model(x + delta)
        loss = loss_fct(y_p, y)
        loss.backward()
        delta.data = (delta + alpha * delta.grad.detach().sign()).clamp(-epsilon, epsilon)
        delta.grad.zero_()
    return delta.detach()

def norms(Z):
    """Compute norms over all but the first dimension"""
    return Z.view(Z.shape[0], -1).norm(dim=1)[:,None,None,None]
def pgd_l2(model, x, y, loss_fct, epsilon=0.2, alpha=5e-4, num_iter=10, randomize=False):
    """ Construct  adversarial examples on the examples X"""
    if randomize:
        delta = torch.rand_like(x, requires_grad=True)
    else:
        delta = torch.zeros_like(x, requires_grad=True)
    for t in range(num_iter):
        y_p = model(x + delta)
        loss = loss_fct(y_p, y)
        loss.backward()
        delta.data += alpha * delta.grad.detach() / norms(delta.grad.detach())
        delta.data = torch.min(torch.max(delta.detach(), -x), 1 - x)  # clip X+delta to [0,1]
        delta.data *= epsilon / norms(delta.detach()).clamp(min=epsilon)
        delta.grad.zero_()
    return delta.detach()

def fgsm(model, x, y, loss_fct, epsilon=8/255):
    """ Construct FGSM adversarial examples on the examples X"""
    delta = torch.zeros_like(x, requires_grad=True)
    y_p = model(x + delta)
    loss = loss_fct(y_p, y)
    loss.backward()
    return epsilon * delta.grad.detach().sign()

def evaluate(model, evasion_loss, val_dataloader):
    loss = 0
    model.eval()
    with torch.no_grad():
        for data in val_dataloader:
            x, y = data
            x = x.type(torch.float32).cuda()
            y = y.type(torch.float32).cuda()
            y_pred = model(x)
            loss += evasion_loss(y_pred, y).item()
    model.train()
    return loss / len(val_dataloader)

def collision_loss_fn(y_p, margin=90):
    """
    Compute the margin collision loss for a batch of predictions. The loss is designed to penalize
    predictions where the pairwise L1 distance for any individual feature is less than a specified margin.

    Parameters:
        y_p (torch.Tensor): A batch of predictions with shape (batch_size, feature_size).
        margin (int, optional): The margin threshold for each feature in the pairwise L1 distance. Defaults to 1800 for PhotoDNA and 90 for PDQ.

    Returns:
        torch.Tensor: The computed loss.
    """
    batch_size, feature_size = y_p.shape

    # Expand y_p to form all pairs (two different expansions for broadcasting)
    y_p1 = y_p.unsqueeze(1).expand(-1, batch_size, -1)  # Shape (batch_size, batch_size, feature_size)
    y_p2 = y_p.unsqueeze(0).expand(batch_size, -1, -1)  # Shape (batch_size, batch_size, feature_size)

    # Compute pairwise L1 distance using broadcasting for each feature
    pairwise_l1_distance = torch.abs(y_p1 - y_p2).sum(dim=2)  # Shape (batch_size, batch_size)
    # Apply the margin condition to each feature individually
    loss = F.relu(margin - pairwise_l1_distance)  # Apply margin threshold, shape remains (batch_size, batch_size, feature_size)

    # Mask out the diagonal elements since we don't want to calculate loss for them
    # mask = torch.eye(batch_size, dtype=torch.bool, device=y_p.device).unsqueeze(-1)
    mask = torch.eye(batch_size, dtype=torch.bool, device=y_p.device)
    loss = loss * (~mask)

    # Sum all elements and average across non-diagonal pairs and features
    total_loss = loss.sum() / (batch_size * (batch_size - 1))

    return total_loss

def get_opts():
    parser = ArgumentParser()
    parser.add_argument(
        "--data_dir", help="Directory containing train/validation images", type=str, default="data"
    )
    parser.add_argument(
        "--train_data",
        help="Training data, CSV of pairs of (path, base64-encoded hash)",
        type=str,
        # required=True,
        default='coco-train.csv'
    )
    parser.add_argument("--val-data", help="Validation data", default='coco-val.csv',type=str, required=False)
    parser.add_argument("--epochs", help="Training epochs", type=int, default=160)
    parser.add_argument("--data_name", type = str,  default='coco', choices=['coco', 'mnist'])
    parser.add_argument(
        "--output",
        help="Name of model output (without extension)",
        type=str,
        default="saved_models/base_adv/mnist-pdq.pt",
    )
    parser.add_argument("--checkpoint_iter", help="Checkpoint frequency", type=int, default=-1)
    parser.add_argument("--batch_size", help="Batch size", type=int, default=64)
    parser.add_argument("--verbose", help="Print intermediate statistics", action="store_true")
    return parser.parse_args()


def main():
    opts = get_opts()
    lambda_c = 1 ##yc: weight parameter for collision loss previous 1
    adv = True

    if opts.data_name == 'mnist':
        opts.train_data = 'data/mnist/mnist_train.csv'
        opts.val_data = 'data/mnist/mnist_test.csv'
        opts.output = 'saved_models/base_adv/mnist-pdq-ep8-pdg'
        opts.epochs = 20
        resize = 28 ##for resize
        model = resnet(in_ch=1, in_dim=resize)  # PDQ
        margin = 90
        train_data = ImageToHashAugmented_PDQ(opts.train_data, opts.data_dir, resize, 1)
        train_dataloader = DataLoader(
            train_data, batch_size=opts.batch_size, shuffle=True, pin_memory=True
        )

        val_data = ImageToHashAugmented_PDQ(opts.val_data, opts.data_dir, resize, 0) if opts.val_data else None
        val_dataloader = (
            DataLoader(val_data, batch_size=opts.batch_size, shuffle=True, pin_memory=True)
            if val_data
            else None
        )

    else:
        opts.train_data = 'data/coco-train.csv'
        opts.val_data = 'data/coco-val.csv'
        opts.output = 'saved_models/base_adv/coco-photodna-ep8-photodna'
        resize = 64 ##for resize
        model = resnet_v5(num_classes=144, bn=True, input_dim=resize)  # PhotoDNA
        margin = 1800

        train_data = ImageToHashAugmented(opts.train_data, opts.data_dir, resize, 1)
        train_dataloader = DataLoader(
            train_data, batch_size=opts.batch_size, shuffle=True, pin_memory=True
        )

        val_data = ImageToHashAugmented(opts.val_data, opts.data_dir, resize, 0) if opts.val_data else None
        val_dataloader = (
            DataLoader(val_data, batch_size=opts.batch_size, shuffle=True, pin_memory=True)
            if val_data
            else None
        )

    model = model.cuda()

    evasion_loss = nn.L1Loss()
    val_evasion_loss = nn.L1Loss()
    optimizer = optim.Adam(model.parameters(), lr=5e-4)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[15,18], gamma=0.2)


    # train
    losses = []
    val_losses = []
    with tqdm(range(opts.epochs), unit="epoch", total=opts.epochs) as tepochs:
        for epoch in tepochs:
            train_loss = 0
            for data in tqdm(
                train_dataloader, unit="batch", total=len(train_dataloader), leave=False
            ):
                # x, y, ori_index = data
                x, y = data
                optimizer.zero_grad()
                x = x.cuda()
                y = y.cuda()
                if adv == True:
                    delta = pgd_linf(model, x, y, evasion_loss)
                    y_p = model(x+delta)
                else:
                    y_p = model(x)
                e_loss = evasion_loss(y_p, y)
                if lambda_c == 0:
                    loss = e_loss
                else:
                    c_loss = collision_loss_fn(y_p, margin=margin)
                    loss = e_loss + lambda_c*c_loss
                loss.backward()
                optimizer.step()
                train_loss += loss
            train_loss = train_loss.item() / len(train_dataloader)
            # save checkpoint
            if opts.checkpoint_iter > 0 and epoch % opts.checkpoint_iter == 0:
                torch.save(model.state_dict(), "{}-epoch{:d}.pt".format(opts.output, epoch))
            # stats
            if opts.verbose:
                tepochs.clear()
                if val_dataloader:
                    val_loss, hamming, acc = evaluate(model, val_evasion_loss, val_dataloader)
                    print(
                        "Epoch {}, train loss: {:.1f},  val loss: {:.1f}".format(
                            epoch, train_loss, val_loss
                        )
                    )
                else:
                    print("Epoch {}, train loss: {:.1f}".format(epoch, train_loss))
            else:
                if val_dataloader:
                    val_loss = evaluate(model,val_evasion_loss, val_dataloader)
                    tepochs.set_postfix(train_loss=train_loss, val_loss=val_loss)
                else:
                    tepochs.set_postfix(train_loss=train_loss)

            scheduler.step()

            losses.append(train_loss)
            if val_dataloader:
                val_losses.append(val_loss)

            if len(val_losses) > 1 and val_loss <= min(val_losses):
                # save best model
                print(f'Save best at epoch {epoch}.')
                torch.save(model.state_dict(), "{}.pt".format(opts.output))

            else:
                print('This epoch not saved')

    print(f"Train loss: {losses}")
    print(f"Val loss: {val_losses}")


if __name__ == "__main__":
    main()
