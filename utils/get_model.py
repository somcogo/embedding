import numpy as np
import torch

from models.embedding_functionals import GeneralBatchNorm2d, GeneralInstanceNorm2d
from models.assembler import ModelAssembler

def get_model(dataset, model_name, site_number, embed_dim, model_type, norm_layer, ft_emb_vec, cifar=True, feature_dims=None, **kwargs):
    if dataset == 'cifar10':
        num_classes = 10
        in_channels = 3
    elif dataset == 'cifar100':
        num_classes = 100
        in_channels = 3
    elif dataset == 'digits':
        num_classes = 10
        in_channels = 3
    config = get_model_config(model_name, model_type, cifar, feature_dims, norm_layer)
    config['gen_dim'] = embed_dim
    models = []
    for _ in range(site_number):
        model = ModelAssembler(channels=in_channels, num_classes=num_classes, emb_dim=embed_dim, **config)
        models.append(model)
    
    if config['mode'] != 'vanilla':
        mu_init = np.eye(site_number, embed_dim)
        for i, model in enumerate(models):
            init_weight = torch.from_numpy(mu_init[i]).to(torch.float) if ft_emb_vec is None else ft_emb_vec
            model.embedding = torch.nn.Parameter(init_weight)
    return models, num_classes

def get_model_config(model_name, model_type, cifar, feature_dims, norm_layer):
    if model_name == 'resnet18':
        config = {
            'backbone_name':'resnet',
            'layers':[2, 2, 2, 2],
            'cifar':cifar,
        }

    if norm_layer == 'bn':
        config['norm_layer'] = GeneralBatchNorm2d
    elif norm_layer == 'in':
        config['norm_layer'] = GeneralInstanceNorm2d

    config['head_name'] = 'classifier'
    config['feature_dims'] = feature_dims if feature_dims is not None else [64, 128, 256, 512]

    if model_type == 'vanilla':
        config['mode'] = 'vanilla'
    elif model_type == 'embbn3':
        config['mode'] = 'fedbn'
        config['gen_depth'] = 1
        config['gen_affine'] = False
        config['gen_hidden_layer'] = 64
        config['use_repl_bn'] = True
    elif model_type == 'embbn4':
        config['mode'] = 'fedbn'
        config['gen_depth'] = 2
        config['gen_affine'] = False
        config['gen_hidden_layer'] = 64
        config['use_repl_bn'] = True
    elif model_type == 'embbn5':
        config['mode'] = 'fedbn'
        config['gen_depth'] = 2
        config['gen_affine'] = False
        config['gen_hidden_layer'] = 64
        config['use_repl_bn'] = True
        config['comb_gen'] = True
    elif model_type == 'embbn6':
        config['mode'] = 'fedbn'
        config['gen_depth'] = 2
        config['gen_affine'] = False
        config['gen_hidden_layer'] = 512
        config['use_repl_bn'] = True
    elif model_type == 'embbn7':
        config['mode'] = 'fedbn'
        config['gen_depth'] = 4
        config['gen_affine'] = False
        config['gen_hidden_layer'] = 64
        config['use_repl_bn'] = True
    elif model_type == 'embbn8':
        config['mode'] = 'fedbn'
        config['gen_depth'] = 3
        config['gen_affine'] = False
        config['gen_hidden_layer'] = 64
        config['use_repl_bn'] = True
    elif model_type == 'embbn9':
        config['mode'] = 'fedbn'
        config['gen_depth'] = 2
        config['gen_affine'] = False
        config['gen_hidden_layer'] = 16
        config['use_repl_bn'] = True
    elif model_type == 'embbn10':
        config['mode'] = 'fedbn'
        config['gen_depth'] = 2
        config['gen_affine'] = False
        config['gen_hidden_layer'] = 256
        config['use_repl_bn'] = True

    return config