#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
import torch.nn.functional as F
from torch.autograd import Variable
from math import exp

def get_depth_loss_fn(loss_name):
    if loss_name == "hinge":
        return hinge_loss
    elif loss_name == "logistic":
        return logistic_loss
    else:
        raise ValueError(f"Unknown depth loss function: {loss_name}")
    
def hinge_loss(front_depth, back_depth):
    return torch.clamp(front_depth - back_depth, min=0).mean()

def logistic_loss(front_depth, back_depth):
    return torch.nn.functional.softplus(front_depth - back_depth).mean()

def soft_opacity_loss(opacity_front, opacity_back):
    """
    Computes a soft loss between two tensors of opacities by calculating
    the Wasserstein distance between their distributions.

    Args:
        opacity_front (torch.Tensor): Opacities from the front layer (B, N, 1).
        opacity_back (torch.Tensor): Opacities from the back layer (B, N, 1).

    Returns:
        torch.Tensor: The computed loss value.
    """
    # Ensure the tensors are 1D by flattening the batch and spatial dimensions
    opacity_front = opacity_front.view(-1)
    opacity_back = opacity_back.view(-1)

    # Sort the opacities to create ordered distributions
    front_sorted, _ = torch.sort(opacity_front)
    back_sorted, _ = torch.sort(opacity_back)

    # Compute the cumulative distribution functions (CDFs)
    front_cdf = torch.cumsum(front_sorted, dim=0)
    back_cdf = torch.cumsum(back_sorted, dim=0)

    # Normalize the CDFs to range [0, 1]
    front_cdf /= front_cdf[-1] + 1e-8  # Add epsilon to avoid division by zero
    back_cdf /= back_cdf[-1] + 1e-8

    # Compute the Wasserstein distance (Earth Mover's Distance)
    loss = torch.mean(torch.abs(front_cdf - back_cdf))

    return loss

def l1_loss(network_output, gt):
    return torch.abs((network_output - gt)).mean()

def l2_loss(network_output, gt):
    return ((network_output - gt) ** 2).mean()

def huber_loss(opacity1, opacity2):
    return torch.nn.functional.huber_loss(opacity1, opacity2).mean()

def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()

def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window

def ssim(img1, img2, window_size=11, size_average=True):
    channel = img1.size(-3)
    window = create_window(window_size, channel)

    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)

    return _ssim(img1, img2, window, window_size, channel, size_average)

def _ssim(img1, img2, window, window_size, channel, size_average=True):
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)

