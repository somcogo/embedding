import torch
from torchvision.transforms import (
     Compose,
     Pad,
     RandomCrop,
     RandomHorizontalFlip,
     RandomApply,
     RandomRotation,
     RandomErasing)

def aug_image(batch: torch.Tensor):
    batch = aug_crop_rotate_flip_erase(batch)
    return batch

def aug_crop_rotate_flip_erase(batch):
    trans = Compose([
        Pad(4),
        RandomCrop(32),
        RandomHorizontalFlip(p=0.25),
        RandomApply(torch.nn.ModuleList([
            RandomRotation(degrees=15)
        ]), p=0.25),
        RandomErasing(p=0.5, scale=(0.015625, 0.25), ratio=(0.25, 4))
    ])
    batch = trans(batch)
    return batch