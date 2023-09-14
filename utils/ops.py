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
    rng = np.random.default_rng()
    if site_id == 0:
        B, H, W, C = batch.shape
        mean = 0
        var = 0.1
        sigma = var**0.5
        gauss = rng.normal(mean, sigma, (B, H, W, C))
        batch = batch + gauss
        return batch
    elif site_id == 1:
        B, H, W, C = batch.shape
        p = 0.002
        salt = rng.binomial(1, p, size=(B, H, W, C))
        pepper = rng.binomial(1, p, size=(B, H, W, C))
        b_min = torch.amin(batch, dim=(1, 2))
        b_max = torch.amax(batch, dim=(1, 2))
        batch[salt] = b_max
        batch[pepper] = b_min
        return batch
    elif site_id == 2:
        B, H, W, C = batch.shape
        print(batch.shape)
        b_max = torch.amax(batch, dim=(1, 2))
        batch = b_max.reshape(B, C, H, W) - batch
        return batch.reshape(B, H, W, C)
    elif site_id == 3:
        B, H, W, C = batch.shape
        batch = F.invert(batch.reshape(B, C, H, W))
        return batch.reshape(B, H, W, C)
    elif site_id == 4:
        B, H, W, C = batch.shape
        g_scale = Grayscale(C)
        batch = g_scale(batch.reshape(B, C, H, W))
        return batch.reshape(B, H, W, C)
    elif site_id == 5:
        color_jitter = ColorJitter()
        batch = color_jitter(batch)
        return batch