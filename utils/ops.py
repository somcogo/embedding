import numpy as np
import torch
from torchvision.transforms import (
     Compose,
     Pad,
     RandomCrop,
     RandomHorizontalFlip,
     RandomVerticalFlip,
     RandomApply,
     RandomRotation,
     RandomErasing,
     RandomResizedCrop,
     ColorJitter,
     Grayscale)
from torchvision.transforms import functional as F

def aug_image(batch: torch.Tensor, dataset):
    batch = aug_crop_rotate_flip_erase(batch, dataset)
    return batch

def aug_crop_rotate_flip_erase(batch, dataset):
    if dataset in ['mnist', 'cifar10', 'cifar100']:
        trans = Compose([
            Pad(4),
            RandomCrop(32),
            RandomHorizontalFlip(p=0.25),
            RandomApply(torch.nn.ModuleList([
                RandomRotation(degrees=15)
            ]), p=0.25),
            RandomErasing(p=0.5, scale=(0.015625, 0.25), ratio=(0.25, 4))
        ])
    elif dataset == 'imagenet':
        trans = Compose([
            Pad(4),
            RandomCrop(64),
            RandomHorizontalFlip(p=0.25),
            RandomApply(torch.nn.ModuleList([
                RandomRotation(degrees=15)
            ]), p=0.25),
            RandomErasing(p=0.5, scale=(0.015625, 0.25), ratio=(0.25, 4))
        ])
    batch = trans(batch)
    return batch

def perturb(batch: torch.Tensor, site_id):
    B, C, H, W = batch.shape
    rng = np.random.default_rng()
    if site_id == 0:
        mean = 0
        var = 0.01
        sigma = var**0.5
        gauss = rng.normal(mean, sigma, (B, C, H, W))
        batch = batch + gauss
        return batch
    elif site_id == 1:
        out = batch
        p = 0.01
        salt = rng.binomial(1, p, size=(B, C, H, W))
        pepper = rng.binomial(1, p, size=(B, C, H, W))
        salt = torch.tensor(salt, dtype=torch.bool)
        pepper = torch.tensor(pepper, dtype=torch.bool)
        b_min = torch.amin(batch, dim=(2, 3), keepdim=True)
        b_max = torch.amax(batch, dim=(2, 3), keepdim=True)
        out = (b_max - out)*salt + out
        out = (b_min - out)*pepper + out
        return out
    elif site_id == 2:
        batch = F.invert(batch)
        return batch
    elif site_id == 3:
        g_scale = Grayscale(C)
        batch = g_scale(batch)
        return batch
    elif site_id == 4:
        color_jitter = ColorJitter(0.5, 0.5, 0.5, 0.5)
        batch = color_jitter(batch)
        return batch