import math
import numpy as np
import torch
from torchvision.transforms import (
     Compose,
     Pad,
     RandomCrop,
     RandomHorizontalFlip,
     RandomApply,
     RandomRotation,
     RandomErasing,
     ColorJitter,
     ConvertImageDtype,
     Resize,
     InterpolationMode,
     v2)
from timm.data.mixup import Mixup

def get_class_list(task, site_number, class_number, class_seed, degradation):
    rng = np.random.default_rng(seed=class_seed)
    if task == 'segmentation' and degradation == 'nothing':
        class_range = rng.permutation(class_number)
        classes = np.array_split(class_range, site_number)
    else:
        classes = [None for i in range(site_number)]

    return classes

def transform_image(batch, labels, mode, transform, dataset, model, p=False, trn_log=True):
    if mode == 'trn':
        batch, labels = aug_image(batch, labels, dataset, model, p=p, trn_log=trn_log)
    if transform is not None:
        # For some transforms, e.g. colorjitter, the pixelvalues need to be in [0, 1]
        b_max = torch.amax(batch, dim=(1, 2, 3), keepdim=True)
        b_min = torch.amin(batch, dim=(1, 2, 3), keepdim=True)
        batch = (batch - b_min) / (b_max - b_min + 1e-5)
        batch = transform(batch)
        batch = (b_max - b_min) * batch + b_min
    if model == 'swinv2':
        resize_tr = Resize((224, 224), antialias=True)
        batch = resize_tr(batch)
        if dataset in ['celeba', 'minicoco']:
            resize_mask = Resize((224, 224), interpolation=InterpolationMode.NEAREST)
            labels = resize_mask(labels)
    return batch, labels

def create_mask_from_onehot(one_hot_mask, classes):
    one_hot_mask = one_hot_mask > 0
    one_hot_mask = one_hot_mask * torch.arange(1, 19, device=one_hot_mask.device)
    mask = (one_hot_mask[..., classes]).amax(dim=-1)
    return mask

def aug_image(batch, labels, dataset, model, p=False, trn_log=True):
    if dataset in ['mnist', 'cifar10', 'cifar100']:
        if p:
            print('aug for mnist, cifar')
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
    elif dataset == 'imagenet' and trn_log:
        if p:
            print('standard aug for imagenet')
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
    elif dataset in ['celeba', 'minicoco']:
        if p:
            print('aug for celeba')
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
    elif dataset == 'imagenet' and not trn_log:
        if p:
            print('new aug for imagenet')
        rand_aug = v2.RandAugment(num_ops=2)
        mixup = Mixup(
            mixup_alpha=0.8,
            cutmix_alpha=1.,
            cutmix_minmax=None,
            prob=0.4,
            switch_prob=0.5,
            mode='elem',
            label_smoothing=0.1,
            num_classes=200
        )
        rand_erase = v2.RandomErasing()
        
        batch = rand_aug(batch)
        batch, labels = mixup(batch, labels)
        batch = rand_erase(batch)
    return batch, labels
    
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
            transforms.append(deterministicColorjitter(bri, con, sat, hue))

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
                
        if site_number > 5:
            transforms = rng.permutation(transforms)

    elif degradation == 'addgauss':
        var_add = np.linspace(kwargs['var_add'][0], kwargs['var_add'][1], site_number)
        for var in var_add:
            transforms.append(NoiseTransform(rng=rng, t_rng=t_rng, device=device, var_add=var, choice=0))
        transforms = rng.permutation(transforms)

    elif degradation == 'nothing' or degradation == 'classsep':
        for site in range(site_number):
            transforms.append(ConvertImageDtype(torch.float))
            
    return transforms

def get_test_transforms(site_number, seed, degradation, device, **kwargs):
    rng = np.random.default_rng(seed)
    transforms = []
    if degradation == 'mixed':
        variances = np.linspace(kwargs['var_add'][0], kwargs['var_add'][1], site_number//2)
        for i in range(site_number//2):
            transforms.append(NoiseTransform(rng=rng, t_rng=None, device=device, var_add=variances[i], choice=0))
        for i in range(site_number//2):
            transforms.append(ConvertImageDtype(torch.float))
    elif degradation == '3mixed':
        variances = np.linspace(kwargs['var_add'][0], kwargs['var_add'][1], site_number//3)
        for i in range(site_number//3):
            transforms.append(NoiseTransform(rng=rng, t_rng=None, device=device, var_add=variances[i], choice=0))
        for i in range(site_number//3):
            transforms.append(ConvertImageDtype(torch.float))
        jitters = np.linspace(0.5, 1.5, site_number//3)
        for j in jitters:
            transforms.append(deterministicColorjitter(j, j, j, j - 1))
    elif degradation == 'addgauss':
        variances = np.linspace(kwargs['var_add'][0], kwargs['var_add'][1], site_number)
        for i in range(site_number):
            transforms.append(NoiseTransform(rng=rng, t_rng=None, device=device, var_add=variances[i], choice=0))
    elif degradation == 'jittermix':
        bri = np.linspace(0.5, 1.5, site_number//4)
        for b in bri:
            transforms.append(deterministicColorjitter(b, 1., 1., 0.))
        con = np.linspace(0.5, 1.5, site_number//4)
        for c in con:
            transforms.append(deterministicColorjitter(1., c, 1., 0.))
        sat = np.linspace(0.5, 1.5, site_number//4)
        for s in sat:
            transforms.append(deterministicColorjitter(1., 1., s, 0.))
        hue = np.linspace(-0.5, 0.5, site_number//4)
        for h in hue:
            transforms.append(deterministicColorjitter(1., 1., 1., h))

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

class deterministicColorjitter:
    def __init__(self, bri, con, sat, hue):
        self.bri = bri
        self.con = con
        self.sat = sat
        self.hue = hue
        self.jitter = ColorJitter((self.bri, self.bri), (self.con, self.con), (self.sat, self.sat), (self.hue, self.hue))
    
    def __call__(self, img:torch.Tensor):
        return self.jitter(img)