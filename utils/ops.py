from typing import Any
import math
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
     Grayscale,
     ConvertImageDtype,)
from torchvision.transforms import functional as F

def get_class_list(task, site_number, class_number, class_seed):
    rng = np.random.default_rng(seed=class_seed)
    if task == 'segmentation':
        class_range = rng.permutation(class_number)
        classes = np.array_split(class_range, site_number)
    else:
        classes = [None for i in range(site_number)]

    return classes

def aug_image(batch: torch.Tensor, labels, dataset):
    batch = aug_crop_rotate_flip_erase(batch, labels, dataset)
    return batch

def aug_crop_rotate_flip_erase(batch, labels, dataset):
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
        batch = trans(batch)
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
    elif dataset == 'celeba':
        flip = torch.rand(size=[batch.shape[0]]) < 0.5
        rotation = torch.randint(0, 5, [batch.shape[0]])
        scales = torch.rand(size=[batch.shape[0]], device=batch.device) * 0.4 + 0.8
        for ndx in range(batch.shape[0]):
            if flip[ndx]:
                batch[ndx] = torch.flip(batch[ndx], dims=(-2, -1))
                labels[ndx] = torch.flip(labels[ndx], dims=(-2, -1))
            batch[ndx] = torch.rot90(batch[ndx], rotation[ndx], dims=(-2, -1))
            labels[ndx] = torch.rot90(labels[ndx], rotation[ndx], dims=(-2, -1))
        batch = scales.reshape(-1, 1, 1, 1) * batch
    return batch, labels

def perturb_colorjitter(batch, site_id):
    rng = np.random.default_rng(seed=site_id)
    brightness = rng.random() + 0.5
    contrast = rng.random() + 0.5
    saturation = rng.random() + 0.5
    hue = rng.random() - 0.5
    color_jitter = ColorJitter((brightness, brightness), (contrast, contrast), (saturation, saturation), (hue, hue))
    return color_jitter(batch)

def perturb_default(batch: torch.Tensor, site_id, device):
    B, C, H, W = batch.shape
    rng = np.random.default_rng()
    if site_id == 0:
        mean = 0
        var = 0.01
        sigma = var**0.5
        gauss = rng.normal(mean, sigma, (B, C, H, W))
        gauss = torch.tensor(gauss, device=device)
        batch = batch + gauss
        return batch.float()
    elif site_id == 1:
        out = batch
        p = 0.01
        salt = rng.binomial(1, p, size=(B, C, H, W))
        pepper = rng.binomial(1, p, size=(B, C, H, W))
        salt = torch.tensor(salt, dtype=torch.bool, device=device)
        pepper = torch.tensor(pepper, dtype=torch.bool, device=device)
        b_min = torch.amin(batch, dim=(2, 3), keepdim=True)
        b_max = torch.amax(batch, dim=(2, 3), keepdim=True)
        out = (b_max - out)*salt + out
        out = (b_min - out)*pepper + out
        return out.float()
    elif site_id == 2:
        batch = F.invert(batch)
        return batch.float()
    elif site_id == 3:
        g_scale = Grayscale(C)
        batch = g_scale(batch)
        return batch.float()
    elif site_id == 4:
        color_jitter = ColorJitter((1.2, 1.2), (.7, .7),(2, 2),(.25, .25),)
        batch = color_jitter(batch)
        return batch.float()
    
def perturb(batch, site_id, device, mode):
    if mode == 'default':
        return perturb_default(batch, site_id, device)
    elif mode == 'colorjitter':
        return perturb_colorjitter(batch, site_id)
    
def principle_comp_analysis(data:torch.Tensor):
    
    centered_data = data - data.mean(dim=1, keepdim=True)
    cov = torch.cov(centered_data.transpose(0, 1))
    evalue, evector = torch.linalg.eigh(cov)
    transformed_data = torch.matmul(centered_data, evector)

    return transformed_data, evector, evalue

def getTransformList(degradation, site_number, seed, device, **kwargs):
    transforms = []
    rng = np.random.default_rng(seed)
    t_rng = torch.Generator(device=device).manual_seed(seed)
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
            transforms.append(ColorJitter((bri, bri),
                                          (con, con),
                                          (sat, sat),
                                          (hue, hue)))

    elif degradation == 'colorjitternoclip':
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
            transforms.append(ColorjitterWithoutClip(bri, con, sat, hue))

    elif degradation == '3noises':
        var_add = np.linspace(kwargs['var_add'][0], kwargs['var_add'][1], math.ceil(site_number/3))
        var_mul = np.linspace(kwargs['var_mul'][0], kwargs['var_mul'][1], math.ceil(site_number/3))
        alphas = np.linspace(kwargs['alpha'][1], kwargs['alpha'][0], site_number - len(var_add) - len(var_mul))
        for ndx in range(len(var_add)):
            if ndx < len(var_add):
                transforms.append(NoiseTransform(rng=rng, t_rng=t_rng, device=device, var_add=var_add[ndx], choice=0))
            if ndx < len(var_mul):
                transforms.append(NoiseTransform(rng=rng, t_rng=t_rng, device=device, var_mul=var_mul[ndx], choice=1))
            if ndx < len(alphas):
                transforms.append(NoiseTransform(rng=rng, t_rng=t_rng, device=device, alpha=alphas[ndx], choice=2))

        # for var in var_add:
        #     transforms.append(NoiseTransform(rng=rng, device=device, var_add=var, choice=0))
        # for var in var_mul:
        #     transforms.append(NoiseTransform(rng=rng, device=device, var_mul=var, choice=1))
        # for alpha in alphas:
        #     transforms.append(NoiseTransform(rng=rng, device=device, alpha=alpha, choice=2))
        if site_number > 5:
            transforms = rng.permutation(transforms)

    elif degradation == 'addgauss':
        var_add = np.linspace(kwargs['var_add'][0], kwargs['var_add'][1], site_number)
        for var in var_add:
            transforms.append(NoiseTransform(rng=rng, t_rng=t_rng, device=device, var_add=var, choice=0))
        transforms = rng.permutation(transforms)

    elif degradation == 'patchswap':
        indices = rng.permutation(np.arange(site_number))
        swap_count = np.arange(1, site_number+1) * kwargs['swap_count']
        patch_size = np.arange(site_number, 0, -1) * kwargs['patch_size']
        for site in range(site_number):
            transforms.append(PatchSwap(rng=rng, patch_size=patch_size[indices[site]], swap_count=swap_count[indices[site]]))

    elif degradation == 'nothing':
        for site in range(site_number):
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
        elif self.kwargs['choice'] == 1:
            sigma = self.kwargs['var_mul']**0.5
            gauss = self.rng.normal(1, sigma, img.shape)
            gauss = torch.tensor(gauss, device=self.device)
            img = img * gauss
        elif self.kwargs['choice'] == 2:
            alpha = self.kwargs['alpha']
            noise = torch.poisson(input=img*10**alpha, generator=self.t_rng)/10**alpha
            img = torch.tensor(noise, device=self.device)
        return img.float()

class PatchSwap:
    def __init__(self, rng, patch_size, swap_count):
        self.rng = rng
        self.patch_size = patch_size
        self.swap_count = swap_count

    def __call__(self, img:torch.Tensor):
        B, C, H, W = img.shape
        for _ in range(self.swap_count):
            for batch_i in range(B):
                overlap = True
                while overlap:
                    x1, x2 = self.rng.integers(0, H - self.patch_size, size=(2))
                    y1, y2 = self.rng.integers(0, W - self.patch_size, size=(2))
                    if (x1 -x2) > self.patch_size or (y1 - y2) > self.patch_size:
                        overlap = False

                patch = img[batch_i, :, x1:x1 + self.patch_size, y1:y1 + self.patch_size].clone()
                img[batch_i, :, x1:x1 + self.patch_size, y1:y1 + self.patch_size] = img[batch_i, :, x2:x2 + self.patch_size, y2:y2 + self.patch_size]
                img[batch_i, :, x2:x2 + self.patch_size, y2:y2 + self.patch_size] = patch
        return img

class ColorjitterWithoutClip:
    def __init__(self, bri, con, sat, hue):
        self.bri = bri
        self.con = con
        self.sat = sat
        self.hue = hue

    def __call__(self, img:torch.Tensor):
        return do_colorjitter(img, self.bri, self.con, self.sat, self.hue)


# TODO: based on the PyTorch Colorjitter implementation
def do_colorjitter(img, bri, con, sat, hue):
    img = blend_without_clip(img, torch.zeros_like(img), bri)

    dtype = img.dtype if torch.is_floating_point(img) else torch.float32
    mean = torch.mean(rgb_to_grayscale(img).to(dtype), dim=(-3, -2, -1), keepdim=True)
    img = blend_without_clip(img, mean, con)

    img = blend_without_clip(img, rgb_to_grayscale(img), sat)


    orig_dtype = img.dtype
    img = img.to(torch.float32)

    hsv_img = _rgb2hsv(img)
    h, s, v = hsv_img.unbind(dim=-3)
    h = (h + hue) % 1.0
    hsv_img = torch.stack((h, s, v), dim=-3)
    img_hue_adj = _hsv2rgb(hsv_img)
    img = img_hue_adj.to(orig_dtype)

    return img

def blend_without_clip(img1: torch.Tensor, img2: torch.Tensor, ratio: float) -> torch.Tensor:
    ratio = float(ratio)
    return (ratio * img1 + (1.0 - ratio) * img2).to(img1.dtype)

def rgb_to_grayscale(img: torch.Tensor, num_output_channels: int = 1) -> torch.Tensor:
    if img.ndim < 3:
        raise TypeError(f"Input image tensor should have at least 3 dimensions, but found {img.ndim}")

    if num_output_channels not in (1, 3):
        raise ValueError("num_output_channels should be either 1 or 3")

    if img.shape[-3] == 3:
        r, g, b = img.unbind(dim=-3)
        # This implementation closely follows the TF one:
        # https://github.com/tensorflow/tensorflow/blob/v2.3.0/tensorflow/python/ops/image_ops_impl.py#L2105-L2138
        l_img = (0.2989 * r + 0.587 * g + 0.114 * b).to(img.dtype)
        l_img = l_img.unsqueeze(dim=-3)
    else:
        l_img = img.clone()

    if num_output_channels == 3:
        return l_img.expand(img.shape)

    return l_img

def _rgb2hsv(img: torch.Tensor) -> torch.Tensor:
    r, g, b = img.unbind(dim=-3)

    # Implementation is based on https://github.com/python-pillow/Pillow/blob/4174d4267616897df3746d315d5a2d0f82c656ee/
    # src/libImaging/Convert.c#L330
    maxc = torch.max(img, dim=-3).values
    minc = torch.min(img, dim=-3).values

    # The algorithm erases S and H channel where `maxc = minc`. This avoids NaN
    # from happening in the results, because
    #   + S channel has division by `maxc`, which is zero only if `maxc = minc`
    #   + H channel has division by `(maxc - minc)`.
    #
    # Instead of overwriting NaN afterwards, we just prevent it from occurring, so
    # we don't need to deal with it in case we save the NaN in a buffer in
    # backprop, if it is ever supported, but it doesn't hurt to do so.
    eqc = maxc == minc

    cr = maxc - minc
    # Since `eqc => cr = 0`, replacing denominator with 1 when `eqc` is fine.
    ones = torch.ones_like(maxc)
    s = cr / torch.where(eqc, ones, maxc)
    # Note that `eqc => maxc = minc = r = g = b`. So the following calculation
    # of `h` would reduce to `bc - gc + 2 + rc - bc + 4 + rc - bc = 6` so it
    # would not matter what values `rc`, `gc`, and `bc` have here, and thus
    # replacing denominator with 1 when `eqc` is fine.
    cr_divisor = torch.where(eqc, ones, cr)
    rc = (maxc - r) / cr_divisor
    gc = (maxc - g) / cr_divisor
    bc = (maxc - b) / cr_divisor

    hr = (maxc == r) * (bc - gc)
    hg = ((maxc == g) & (maxc != r)) * (2.0 + rc - bc)
    hb = ((maxc != g) & (maxc != r)) * (4.0 + gc - rc)
    h = hr + hg + hb
    h = torch.fmod((h / 6.0 + 1.0), 1.0)
    return torch.stack((h, s, maxc), dim=-3)


def _hsv2rgb(img: torch.Tensor) -> torch.Tensor:
    h, s, v = img.unbind(dim=-3)
    i = torch.floor(h * 6.0)
    f = (h * 6.0) - i
    i = i.to(dtype=torch.int32)

    p = torch.clamp((v * (1.0 - s)), 0.0, 1.0)
    q = torch.clamp((v * (1.0 - s * f)), 0.0, 1.0)
    t = torch.clamp((v * (1.0 - s * (1.0 - f))), 0.0, 1.0)
    i = i % 6

    mask = i.unsqueeze(dim=-3) == torch.arange(6, device=i.device).view(-1, 1, 1)

    a1 = torch.stack((v, q, p, p, t, v), dim=-3)
    a2 = torch.stack((t, v, v, q, p, p), dim=-3)
    a3 = torch.stack((p, p, t, v, v, q), dim=-3)
    a4 = torch.stack((a1, a2, a3), dim=-4)

    return torch.einsum("...ijk, ...xijk -> ...xjk", mask.to(dtype=img.dtype), a4)

