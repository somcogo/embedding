from models.embedding_functionals import GeneralBatchNorm2d

def get_model_config(model_name, model_type, task):
    if model_name == 'resnet18':
        config = {
            'backbone_name':'resnet',
            'layers':[2, 2, 2, 2],
            'norm_layer':GeneralBatchNorm2d,
            'cifar':True,
        }
    elif model_name == 'internimage':
        config = {}

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