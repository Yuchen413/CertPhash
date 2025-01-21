import torch
def l_infinity_loss(predicted, target):
    return torch.norm(predicted - target, p=float('inf'))