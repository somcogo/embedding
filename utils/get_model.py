from models.model import ResNet18Model, ResNet34Model, ResNetWithEmbeddings, CustomResnet
from models.embedding_functionals import GeneralInstanceNorm2d, BatchNorm2d_noemb, GeneralBatchNorm2d
# from models.maxvit import MaxViT
# from models.maxvitemb import MaxViTEmb

def get_model(dataset, model_name, site_number, embed_dim=None, layer_number=None, pretrained=False, conv1_residual=True, fc_residual=True, model_type=None, cifar=True, extra_conv=False):
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
    models = []
    for _ in range(site_number):
        if model_name == 'resnet34emb':
            model = ResNetWithEmbeddings(num_classes=num_classes, in_channels=in_channels, layers=[3, 4, 6, 3], site_number=site_number, embed_dim=embed_dim, layer_number=layer_number, conv1_residual=conv1_residual, fc_residual=fc_residual)
        elif model_name == 'resnet18emb':
            model = ResNetWithEmbeddings(num_classes=num_classes, in_channels=in_channels, layers=[2, 2, 2, 2], site_number=site_number, embed_dim=embed_dim, layer_number=layer_number, conv1_residual=conv1_residual, fc_residual=fc_residual)
        elif model_name == 'resnet18embhypnn1':
            model = ResNetWithEmbeddings(num_classes=num_classes, in_channels=in_channels, layers=[2, 2, 2, 2], site_number=site_number, embed_dim=embed_dim, use_hypnns=True, version=1, layer_number=layer_number, conv1_residual=conv1_residual, fc_residual=fc_residual)
        elif model_name == 'resnet18embhypnn2':
            model = ResNetWithEmbeddings(num_classes=num_classes, in_channels=in_channels, layers=[2, 2, 2, 2], site_number=site_number, embed_dim=embed_dim, use_hypnns=True, version=2, layer_number=layer_number, conv1_residual=conv1_residual, fc_residual=fc_residual)
        elif model_name == 'resnet18lightweight1':
            model = ResNetWithEmbeddings(num_classes=num_classes, in_channels=in_channels, layers=[2, 2, 2, 2], site_number=site_number, embed_dim=embed_dim, use_hypnns=True, version=1, lightweight=True, layer_number=layer_number, conv1_residual=conv1_residual, fc_residual=fc_residual)
        elif model_name == 'resnet18lightweight2':
            model = ResNetWithEmbeddings(num_classes=num_classes, in_channels=in_channels, layers=[2, 2, 2, 2], site_number=site_number, embed_dim=embed_dim, use_hypnns=True, version=2, lightweight=True, layer_number=layer_number, conv1_residual=conv1_residual, fc_residual=fc_residual)
        elif model_name == 'resnet18affine1':
            model = ResNetWithEmbeddings(num_classes=num_classes, in_channels=in_channels, layers=[2, 2, 2, 2], site_number=site_number, embed_dim=embed_dim, use_hypnns=True, version=1, lightweight=True, affine=True, layer_number=layer_number, conv1_residual=conv1_residual, fc_residual=fc_residual)
        elif model_name == 'resnet18affine2':
            model = ResNetWithEmbeddings(num_classes=num_classes, in_channels=in_channels, layers=[2, 2, 2, 2], site_number=site_number, embed_dim=embed_dim, use_hypnns=True, version=2, lightweight=True, affine=True, layer_number=layer_number, conv1_residual=conv1_residual, fc_residual=fc_residual)
        elif model_name == 'resnet18medium1':
            model = ResNetWithEmbeddings(num_classes=num_classes, in_channels=in_channels, layers=[2, 2, 2, 2], site_number=site_number, embed_dim=embed_dim, use_hypnns=True, version=1, lightweight=True, affine=True, medium_ffwrd=True, layer_number=layer_number, conv1_residual=conv1_residual, fc_residual=fc_residual)
        elif model_name == 'resnet18medium2':
            model = ResNetWithEmbeddings(num_classes=num_classes, in_channels=in_channels, layers=[2, 2, 2, 2], site_number=site_number, embed_dim=embed_dim, use_hypnns=True, version=2, lightweight=True, affine=True, medium_ffwrd=True, layer_number=layer_number, conv1_residual=conv1_residual, fc_residual=fc_residual)
        elif model_name == 'resnet34':
            model = ResNet34Model(num_classes=num_classes, in_channels=in_channels, pretrained=pretrained)
        elif model_name == 'resnet18':
            if model_type == 'vanilla_old':
                model = ResNet18Model(num_classes=num_classes, in_channels=in_channels, cifar=cifar)
            elif model_type == 'emb_old':
                model = ResNetWithEmbeddings(num_classes=num_classes, in_channels=in_channels, layers=[2, 2, 2, 2], site_number=site_number, embed_dim=embed_dim, layer_number=layer_number, conv1_residual=conv1_residual, fc_residual=fc_residual, cifar=cifar)
            elif model_type == 'lightweight_old':
                model = ResNetWithEmbeddings(num_classes=num_classes, in_channels=in_channels, layers=[2, 2, 2, 2], site_number=site_number, embed_dim=embed_dim, use_hypnns=True, version=1, lightweight=True, layer_number=layer_number, conv1_residual=conv1_residual, fc_residual=fc_residual, cifar=cifar)
            else:
                weight_gen_args, mode, norm_layer = get_gen_args(model_type, embed_dim)
                model = CustomResnet(num_classes=num_classes, in_channels=in_channels, mode=mode, weight_gen_args=weight_gen_args, norm_layer=norm_layer, cifar=cifar, extra_conv=extra_conv)
        models.append(model)
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