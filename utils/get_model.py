import numpy as np
import torch
import torch.nn as nn

from models.embedding_functionals import GeneralBatchNorm2d
from models.assembler import ModelAssembler

def get_model(dataset, model_name, site_number, embed_dim=None, model_type=None, task=None, cifar=True, feature_dims=None):
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
        num_classes = 19
        in_channels = 3
    config = get_model_config(model_name, model_type, task, cifar, feature_dims)
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

def get_model_config(model_name, model_type, task, cifar, feature_dims):
    if model_name == 'resnet18':
        config = {
            'backbone_name':'resnet',
            'layers':[2, 2, 2, 2],
            'norm_layer':GeneralBatchNorm2d,
            'cifar':cifar,
        }
    elif model_name == 'convnext':
        config = {
            'backbone_name':'convnext',
            'depths':[3, 3, 9, 3],
            'dims':feature_dims,
            'drop_path_rate':0.1,
            'norm_layer':nn.BatchNorm2d,
            'patch_size':2
        }
    elif model_name == 'convnextog':
        config = {
            'backbone_name':'convnextog',
            'depths':[3, 3, 9, 3],
            'dims':feature_dims,
            'drop_path_rate':0.1,
            'norm_layer':nn.BatchNorm2d,
            'out_indices':[0, 1, 2, 3],
            'patch_size':2
        }
    elif model_name == 'swinv2':
        config = {
            'backbone_name':'swinv2',
            'drop_path_rate':0.2,
            'embed_dim':feature_dims[0]
        }

    if task == 'classification':
        config['head_name'] = 'classifier'
        config['feature_dims'] = feature_dims if feature_dims is not None else [64, 128, 256, 512]
    elif task == 'segmentation':
        config['head_name'] = 'upernet'
        config['fpn_out'] = feature_dims[0] if feature_dims is not None else 64
        config['feature_channels'] = feature_dims if feature_dims is not None else [64, 128, 256, 512]
        config['bin_sizes'] = [1, 2, 4, 6]
        config['feature_dims'] = feature_dims if feature_dims is not None else [64, 128, 256, 512]

    if model_type == 'vanilla':
        config['mode'] = 'vanilla'
    elif model_type == 'embtiny':
        config['mode'] = 'embedding_weights'
        config['gen_size'] = 1
        config['gen_depth'] = 1
        config['gen_affine'] = False
        config['gen_hidden_layer'] = 64
    elif model_type == 'embres1':
        config['mode'] = 'embedding_residual'
        config['gen_size'] = 1
        config['gen_depth'] = 1
        config['gen_affine'] = False
        config['gen_hidden_layer'] = 64
    elif model_type == 'embres2':
        config['mode'] = 'embedding_residual'
        config['gen_size'] = 1
        config['gen_depth'] = 2
        config['gen_affine'] = False
        config['gen_hidden_layer'] = 64

    return config