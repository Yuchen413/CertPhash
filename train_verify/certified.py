import torch
import torch.nn as nn
from utils import *
from auto_LiRPA import BoundedTensor, BoundDataParallel
from auto_LiRPA.perturbations import *
from auto_LiRPA.bound_ops import *
import pdb
import math
import torch.nn.functional as F

# eps is normalized by max_eps
def get_crown_loss(args, lb, eps=None):
    lb_padded = torch.cat([torch.zeros_like(lb[:, :1]), lb], dim=1)
    fake_labels = torch.zeros(lb.size(0), dtype=torch.long, device=lb.device)
    if args.ls > 0:
        threshold = 1 - eps * args.ls
        prob = nn.Softmax(dim=-1)(-lb_padded)[:, 0]
        robust_loss_ = (-torch.log(prob[:]) * (prob < threshold).float()).mean()
        return robust_loss_
    robust_loss_ = ce_loss(-lb_padded, fake_labels)
    return robust_loss_


def l1_loss(y_pred, y_true):
    loss = nn.L1Loss()
    return loss(y_pred, y_true)

def robust_l1_loss(lb, ub, y_true):
    l1 = nn.L1Loss()
    lower_loss = l1(lb, y_true)
    upper_loss = l1(ub, y_true)
    combined_loss = torch.max(lower_loss, upper_loss)
    return torch.mean(combined_loss)



def robust_collision_loss(lower_bound, upper_bound, margin=3000):
    """
    Compute the margin collision loss for a batch of predictions based on their lower and upper bounds.
    The loss is designed to penalize predictions where the pairwise L1 distance for any individual feature
    is less than a specified margin.

    Parameters:
        lower_bound (torch.Tensor): The lower bound of predictions with shape (batch_size, feature_size).
        upper_bound (torch.Tensor): The upper bound of predictions with shape (batch_size, feature_size).
        margin (int, optional): The margin threshold for each feature in the pairwise L1 distance. Defaults to 1800.

    Returns:
        torch.Tensor: The computed loss.
    """
    batch_size, feature_size = lower_bound.shape

    # Expand bounds to form all pairs (two different expansions for broadcasting)
    lb1 = lower_bound.unsqueeze(1).expand(-1, batch_size, -1)  # Shape (batch_size, batch_size, feature_size)
    ub1 = upper_bound.unsqueeze(1).expand(-1, batch_size, -1)
    lb2 = lower_bound.unsqueeze(0).expand(batch_size, -1, -1)
    ub2 = upper_bound.unsqueeze(0).expand(batch_size, -1, -1)

    # Compute pairwise minimum and maximum L1 distances
    min_distance = torch.abs(lb1 - ub2).sum(dim=2)  # Shape (batch_size, batch_size)
    max_distance = torch.abs(ub1 - lb2).sum(dim=2)

    # Choose the smaller of the min and max distances to ensure a conservative estimate of collision
    pairwise_l1_distance = torch.min(min_distance, max_distance)

    # Apply the margin condition
    loss = F.relu(margin - pairwise_l1_distance)  # Apply margin threshold

    # Mask out the diagonal elements since we don't want to calculate loss for them
    mask = torch.eye(batch_size, dtype=torch.bool, device=lower_bound.device)
    loss = loss * (~mask)  # Non-in-place masking

    # Sum all elements and average across non-diagonal pairs
    total_loss = loss.sum() / (batch_size * (batch_size - 1))

    return total_loss


def get_C(args, data, labels):
    return get_spec_matrix(data, labels, args.num_class)

def get_bound_loss(args, model, loss_fusion, eps_scheduler,
                    x=None, data=None, labels = None, eps=None,
                    meter=None, train=False, threshold = 90, margin= 90):
    if loss_fusion:
        c, bound_lower, bound_upper = None, False, True
    else:
        # c, bound_lower, bound_upper = get_C(args, data, labels), True, False
        c, bound_lower, bound_upper = None, True, False
    if args.bound_type == 'IBP':
        # FIXME remove `x=x`???
        lb, ub = model(method_opt="compute_bounds", x=x, IBP=True, C=c, method=None,
                        no_replicas=True)
    elif args.bound_type == 'CROWN-IBP':
        factor = (eps_scheduler.get_max_eps() - eps_scheduler.get_eps()) / eps_scheduler.get_max_eps()
        ilb, iub = model.compute_bounds(IBP=True, C=c, method=None)
        if factor < 1e-5:
            lb, ub = ilb, iub
        else:
            clb, cub = model.compute_bounds(IBP=False, C=c, method='backward', 
                bound_lower=bound_lower, bound_upper=bound_upper)
            # clb, cub, A_dict = model.compute_bounds(IBP=False, C=c, method='backward',
            #      bound_lower=bound_lower, bound_upper=bound_upper, return_A=True)
            if loss_fusion:
                ub = cub * factor + iub * (1 - factor)
            else:
                lb = clb * factor + ilb * (1 - factor)
    else:
        raise ValueError
    update_relu_stat(model, meter)

    if loss_fusion:
        if isinstance(model, BoundDataParallel):
            raise NotImplementedError
        return None, torch.mean(torch.log(ub) + get_exp_module(model).max_input)
    else:
        robust_loss = robust_l1_loss(lb, ub, labels) + 1*robust_collision_loss(lb,ub,margin=margin) #1 is the default weight
        return lb, ub, robust_loss

def cert(args, model, model_ori, epoch, epoch_progress, data, labels, eps, data_max, data_min, std, 
        robust=False, reg=False, loss_fusion=False, eps_scheduler=None,
        train=False, meter=None, threshold = 90):
    if not robust and reg:
        eps = max(eps, args.min_eps_reg)
    if type(eps) == float:
        eps = (eps / std).view(1,-1,1,1)
    else: # [batch_size, channels]
        eps = (eps.view(*eps.shape, 1, 1) / std.view(1, -1, 1, 1))

    data_ub = torch.min(data + eps, data_max)
    data_lb = torch.max(data - eps, data_min) 
    ptb = PerturbationLpNorm(norm=np.inf, eps=eps, x_L=data_lb, x_U=data_ub)
    x = BoundedTensor(data, ptb)

    if 'photodna' in args.dir:
        post_process = HashPostProcessing.post_photodna
        threshold = 1800
        margin = 1800
    elif 'pdq' in args.dir:
        post_process = HashPostProcessing.post_pdq
        threshold = 90
        margin = 90
    else:
        post_process = HashPostProcessing.noop
        threshold = 0
        margin = 0

    if loss_fusion:
        x = (x, labels)
        output = model(*x)
        regular_loss = torch.mean(torch.log(output) + get_exp_module(model).max_input)
        regular_err = None

    else:
        output = model(x)
        regular_loss = l1_loss(output, labels)
        err_loss = nn.L1Loss(reduction='none')
        regular_err = torch.sum(err_loss(post_process(output), labels).sum(1) >= threshold).item() / x.size(0)
        x = (x,)
    if robust or reg or args.xiao_reg or args.vol_reg:
        lb, ub, robust_loss = get_bound_loss(args, model, loss_fusion, eps_scheduler,
            x=(x if loss_fusion else None), data=data, labels=labels, 
            eps=eps, meter=meter, train=train, threshold = threshold, margin=margin)
        err_loss = nn.L1Loss(reduction='none')
        robust_err_lb = torch.sum(err_loss(post_process(lb), labels).sum(1) >= threshold).item()
        robust_err_ub = torch.sum(err_loss(post_process(ub), labels).sum(1) >= threshold).item()
        robust_err = max(robust_err_ub, robust_err_lb) / data.size(0) if not loss_fusion else None

    else:
        robust_loss = robust_err = None
    if robust_loss is not None and torch.isnan(robust_loss):
        robust_err = 100.

    return regular_loss, robust_loss, regular_err, robust_err