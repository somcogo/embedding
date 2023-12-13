import numpy as np
import torch

from models.embedding_functionals import GeneralInstanceNorm2d, BatchNorm2d_noemb, GeneralBatchNorm2d
from models.upernet import ModelAssembler
from utils.config import get_model_config

def get_model(dataset, model_name, site_number, embed_dim=None, model_type=None, task=None, cifar=True, logger=None):
    if dataset == 'cifar10':
        num_classes = 10
        in_channels = 3
    elif dataset == 'cifar100':
        num_classes = 100
        in_channels = 3
    elif dataset == 'pascalvoc':
        num_classes = 21
        in_channels = 3
    elif dataset == 'mnist':
        num_classes = 10
        in_channels = 1
    elif dataset == 'imagenet':
        num_classes = 200
        in_channels = 3
    elif dataset == 'celeba':
        num_classes = 18
        in_channels = 3
    config = get_model_config(model_name, model_type, task, cifar, logger)
    models = []
    for _ in range(site_number):
        model = ModelAssembler(channels=in_channels, num_classes=num_classes, emb_dim=embed_dim, **config)
        models.append(model)
    
    if config['mode'] != 'vanilla':
        if embed_dim > 2:
            mu_init = np.eye(site_number, embed_dim)
        else:
            mu_init = np.exp((2 * np.pi * 1j/ site_number)*np.arange(0,site_number))
            mu_init = np.stack([np.real(mu_init), np.imag(mu_init)], axis=1)
        for i, model in enumerate(models):
            init_weight = torch.from_numpy(mu_init[i])
            model.embedding = torch.nn.Parameter(init_weight)
    return models, num_classes