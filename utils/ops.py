import numpy as np
import torch
from torchvision.transforms import (
     functional,
     Compose,
     Pad,
     RandomCrop,
     RandomHorizontalFlip,
     RandomApply,
     RandomRotation,
     RandomErasing,
     ConvertImageDtype,
     v2)
from timm.data.mixup import Mixup

def refactored_get_ft_indices(site_number, ft_site_number, degs, seed=0):
    part_rng = np.random.default_rng(seed)
    if degs == ['digits']:
        deg_nr = 4
        site_per_deg = site_number // 4
        ft_site_per_deg = ft_site_number // 4
    else:
        deg_nr = len(degs)
        site_per_deg = site_number // len(degs)
        ft_site_per_deg = ft_site_number // len(degs)
    indices = []
    for i in range(deg_nr):
        indices.append(part_rng.permutation(np.arange(i * site_per_deg, (i+1) * site_per_deg))[:ft_site_per_deg])
    indices = np.concatenate(indices)

    return indices

def transform_image(batch, labels, mode, transform):
    if mode == 'trn':
        batch, labels = aug_image(batch, labels)
    if transform is not None:
        # For some transforms, e.g. colorjitter, the pixelvalues need to be in [0, 1]
        b_max = torch.amax(batch, dim=(1, 2, 3), keepdim=True)
        b_min = torch.amin(batch, dim=(1, 2, 3), keepdim=True)
        batch = (batch - b_min) / (b_max - b_min + 1e-5)
        batch = transform(batch)
        batch = (b_max - b_min) * batch + b_min
    return batch, labels

def aug_image(batch, labels):
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
    return batch, labels

def refactored_get_transforms(site_number, seed, degs, device, **kwargs):
    if type(degs) != list:
        degs = [degs]

    assert site_number % len(degs) == 0
    site_per_deg = site_number // len(degs)
    transforms = []
    for i, deg in enumerate(degs):
        deg_transforms = get_test_transforms(site_per_deg, seed, deg, device, **kwargs)
        transforms.extend(deg_transforms)
    return transforms

def get_test_transforms(site_number, seed, degradation, device, **kwargs):
    rng = np.random.default_rng(seed)
    transforms = []
    if degradation == 'colorjitter':
        endpoints = np.linspace(0.5, 1.5, site_number+1)
        bri_ndx = rng.permutation(np.arange(site_number))
        con_ndx = rng.permutation(np.arange(site_number))
        sat_ndx = rng.permutation(np.arange(site_number))
        hue_ndx = rng.permutation(np.arange(site_number))
        for site in range(site_number):
            bri = rng.uniform(endpoints[bri_ndx[site]], endpoints[bri_ndx[site]+1])
            con = rng.uniform(endpoints[con_ndx[site]], endpoints[con_ndx[site]+1])
            sat = rng.uniform(endpoints[sat_ndx[site]], endpoints[sat_ndx[site]+1])
            hue = rng.uniform(endpoints[hue_ndx[site]], endpoints[hue_ndx[site]+1]) - 1
            transforms.append(deterministicColorjitter(bri, con, sat, hue, rng))
    elif degradation == 'addgauss':
        variances = np.linspace(kwargs['var_add'][0], kwargs['var_add'][1], site_number)
        for i in range(site_number):
            transforms.append(NoiseTransform(rng=rng, t_rng=None, device=device, var_add=variances[i], choice=0))
    elif degradation in ['alphascale', 'digits']:
        for i in range(site_number):
            transforms.append(ConvertImageDtype(torch.float))

    return transforms

class NoiseTransform:
    def __init__(self, rng, t_rng, device, **kwargs):
        self.rng = rng
        self.t_rng = t_rng
        self.device = device
        self.kwargs = kwargs

    def __call__(self, img:torch.Tensor):
        if self.kwargs['choice'] == 0:
            sigma = self.kwargs['var_add']**0.5
            gauss = self.rng.normal(0, sigma, img.shape)
            gauss = torch.tensor(gauss, device=self.device)
            img = img + gauss
        return img.float()

class deterministicColorjitter:
    def __init__(self, bri, con, sat, hue, rng):
        self.bri = bri
        self.con = con
        self.sat = sat
        self.hue = hue
        self.rng = rng
    
    def __call__(self, img:torch.Tensor):
        fn_idx = self.rng.permutation(4)
        for fn_id in fn_idx:
            if fn_id == 0:
                img = functional.adjust_brightness(img, self.bri)
            if fn_id == 1:
                img = functional.adjust_contrast(img, self.con)
            if fn_id == 2:
                img = functional.adjust_saturation(img, self.sat)
            if fn_id == 3:
                img = functional.adjust_hue(img, self.hue)
        return img