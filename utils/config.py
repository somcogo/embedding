from models.embedding_functionals import GeneralBatchNorm2d

def get_model_config(model_name, model_type, task, cifar, logger):
    if model_name == 'resnet18':
        config = {
            'backbone_name':'resnet',
            'layers':[2, 2, 2, 2],
            'norm_layer':GeneralBatchNorm2d,
            'cifar':cifar,
        }
    elif model_name == 'internimage':
        config = {
            'backbone_name':'internimage',
            'core_op':'DCNv3',
            'ii_channels':64,
            'depths':[4, 4, 18, 4],
            'groups':[4, 8, 16, 32],
            'mlp_ratio':4.,
            'drop_path_rate':0.2,
            'norm_layer':'BN',
            'layer_scale':1.0,
            'offset_scale':1.0,
            'post_norm':False,
            'with_cp':False,
            'out_indices':(0, 1, 2, 3),
            'logger':logger
        }

    if task == 'classification':
        config['head_name'] = 'classifier'
    elif task == 'segmentation':
        config['head_name'] = 'upernet'
        config['fpn_out'] = 64
        config['feature_channels'] = [64, 128, 256, 512]
        config['bin_sizes'] = [1, 2, 4, 6]

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