import numpy as np
import torch

from models.model import ResNet18Model, ResNet34Model, ResNetWithEmbeddings, CustomResnet
from models.embedding_functionals import GeneralInstanceNorm2d, BatchNorm2d_noemb, GeneralBatchNorm2d
from models.upernet import ModelAssembler
from utils.config import get_model_config

def get_model(dataset, model_name, site_number, embed_dim=None, model_type=None, task=None, cifar=True):
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
    config = get_model_config(model_name, model_type, task, cifar)
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


def get_gen_args(model_type, emb_dim):
    if model_type == 'embv1':
        weight_gen_args = {'emb_dim':emb_dim,
                            'size':2,
                            'gen_depth':2,
                            'gen_affine':False,
                            'hidden_layer':64}
        mode = 'embedding_weights'
        norm_layer = GeneralBatchNorm2d
    elif model_type == 'embtiny':
        weight_gen_args = {'emb_dim':emb_dim,
                            'size':1,
                            'gen_depth':1,
                            'gen_affine':False,
                            'hidden_layer':64}
        mode = 'embedding_weights'
        norm_layer = GeneralBatchNorm2d
    elif model_type == 'embres1':
        weight_gen_args = {'emb_dim':emb_dim,
                            'size':1,
                            'gen_depth':1,
                            'gen_affine':False,
                            'hidden_layer':64}
        mode='embedding_residual'
        norm_layer = GeneralBatchNorm2d
    elif model_type == 'embres2':
        weight_gen_args = {'emb_dim':emb_dim,
                            'size':1,
                            'gen_depth':2,
                            'gen_affine':False,
                            'hidden_layer':64}
        mode='embedding_residual'
        norm_layer = GeneralBatchNorm2d
    elif model_type == 'embv1_bnormnoemb':
        weight_gen_args = {'emb_dim':emb_dim,
                            'size':2,
                            'gen_depth':2,
                            'gen_affine':False,
                            'hidden_layer':64}
        mode = 'embedding_weights'
        norm_layer = BatchNorm2d_noemb
    elif model_type == 'vanilla':
        weight_gen_args = {'emb_dim':None,
                            'size':None,
                            'gen_depth':None,
                            'gen_affine':None,
                            'hidden_layer':None}
        mode = 'vanilla'
        norm_layer = GeneralBatchNorm2d
    if model_type == 'embv2':
        weight_gen_args = {'emb_dim':emb_dim,
                            'size':1,
                            'gen_depth':2,
                            'gen_affine':False,
                            'hidden_layer':64}
        mode = 'embedding_weights'
        norm_layer = GeneralBatchNorm2d
    if model_type == 'embv3':
        weight_gen_args = {'emb_dim':emb_dim,
                            'size':1,
                            'gen_depth':2,
                            'gen_affine':False,
                            'hidden_layer':8}
        mode = 'embedding_weights'
        norm_layer = GeneralBatchNorm2d
    if model_type == 'embv4':
        weight_gen_args = {'emb_dim':emb_dim,
                            'size':1,
                            'gen_depth':2,
                            'gen_affine':False,
                            'hidden_layer':emb_dim}
        mode = 'embedding_weights'
        norm_layer = GeneralBatchNorm2d
    if model_type == 'embv5':
        weight_gen_args = {'emb_dim':emb_dim,
                            'size':1,
                            'gen_depth':2,
                            'gen_affine':True,
                            'hidden_layer':64}
        mode = 'embedding_weights'
        norm_layer = GeneralBatchNorm2d
    if model_type == 'embv6':
        weight_gen_args = {'emb_dim':emb_dim,
                            'size':1,
                            'gen_depth':1,
                            'gen_affine':True,
                            'hidden_layer':64}
        mode = 'embedding_weights'
        norm_layer = GeneralBatchNorm2d
    if model_type == 'embv7':
        weight_gen_args = {'emb_dim':emb_dim,
                            'size':2,
                            'gen_depth':1,
                            'gen_affine':False,
                            'hidden_layer':64}
        mode = 'embedding_weights'
        norm_layer = GeneralBatchNorm2d
    if model_type == 'embv8':
        weight_gen_args = {'emb_dim':emb_dim,
                            'size':2,
                            'gen_depth':2,
                            'gen_affine':False,
                            'hidden_layer':64}
        mode = 'embedding_weights'
        norm_layer = GeneralBatchNorm2d
    if model_type == 'embv9':
        weight_gen_args = {'emb_dim':emb_dim,
                            'size':1,
                            'gen_depth':1,
                            'gen_affine':False,
                            'hidden_layer':128}
        mode = 'embedding_weights'
        norm_layer = GeneralBatchNorm2d
    if model_type == 'embv10':
        weight_gen_args = {'emb_dim':emb_dim,
                            'size':1,
                            'gen_depth':2,
                            'gen_affine':False,
                            'hidden_layer':128}
        mode = 'embedding_weights'
        norm_layer = GeneralBatchNorm2d
    if model_type == 'embv11':
        weight_gen_args = {'emb_dim':emb_dim,
                            'size':2,
                            'gen_depth':2,
                            'gen_affine':False,
                            'hidden_layer':128}
        mode = 'embedding_weights'
        norm_layer = GeneralBatchNorm2d
    if model_type == 'embv12':
        weight_gen_args = {'emb_dim':emb_dim,
                            'size':2,
                            'gen_depth':2,
                            'gen_affine':False,
                            'hidden_layer':256}
        mode = 'embedding_weights'
        norm_layer = GeneralBatchNorm2d
    if model_type == 'embv13':
        weight_gen_args = {'emb_dim':emb_dim,
                            'size':1,
                            'gen_depth':2,
                            'gen_affine':False,
                            'hidden_layer':256}
        mode = 'embedding_weights'
        norm_layer = GeneralBatchNorm2d
    if model_type == 'embv14':
        weight_gen_args = {'emb_dim':emb_dim,
                            'size':1,
                            'gen_depth':2,
                            'gen_affine':False,
                            'hidden_layer':512}
        mode = 'embedding_weights'
        norm_layer = GeneralBatchNorm2d
    if model_type == 'embres5':
        weight_gen_args = {'emb_dim':emb_dim,
                            'size':2,
                            'gen_depth':1,
                            'gen_affine':False,
                            'hidden_layer':256}
        mode = 'embedding_residual'
        norm_layer = GeneralBatchNorm2d
    if model_type == 'embres6':
        weight_gen_args = {'emb_dim':emb_dim,
                            'size':2,
                            'gen_depth':2,
                            'gen_affine':False,
                            'hidden_layer':256}
        mode = 'embedding_residual'
        norm_layer = GeneralBatchNorm2d
    if model_type == 'embres4':
        weight_gen_args = {'emb_dim':emb_dim,
                            'size':2,
                            'gen_depth':2,
                            'gen_affine':False,
                            'hidden_layer':256}
        mode = 'embedding_residual'
        norm_layer = GeneralBatchNorm2d

    return weight_gen_args, mode, norm_layer