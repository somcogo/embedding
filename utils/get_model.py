from models.model import ResNet18Model, ResNet34Model, ResNetWithEmbeddings
from models.maxvit import MaxViT
from models.maxvitemb import MaxViTEmb

def get_model(dataset, model_name, site_number, embed_dim=None, layer_number=None, pretrained=False):
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
            model = ResNetWithEmbeddings(num_classes=num_classes, in_channels=in_channels, layers=[3, 4, 6, 3], site_number=site_number, embed_dim=embed_dim, layer_number=layer_number)
        elif model_name == 'resnet18emb':
            model = ResNetWithEmbeddings(num_classes=num_classes, in_channels=in_channels, layers=[2, 2, 2, 2], site_number=site_number, embed_dim=embed_dim, layer_number=layer_number)
        elif model_name == 'resnet18embhypnn1':
            model = ResNetWithEmbeddings(num_classes=num_classes, in_channels=in_channels, layers=[2, 2, 2, 2], site_number=site_number, embed_dim=embed_dim, use_hypnns=True, version=1, layer_number=layer_number)
        elif model_name == 'resnet18embhypnn2':
            model = ResNetWithEmbeddings(num_classes=num_classes, in_channels=in_channels, layers=[2, 2, 2, 2], site_number=site_number, embed_dim=embed_dim, use_hypnns=True, version=2, layer_number=layer_number)
        elif model_name == 'resnet18lightweight1':
            model = ResNetWithEmbeddings(num_classes=num_classes, in_channels=in_channels, layers=[2, 2, 2, 2], site_number=site_number, embed_dim=embed_dim, use_hypnns=True, version=1, lightweight=True, layer_number=layer_number)
        elif model_name == 'resnet18lightweight2':
            model = ResNetWithEmbeddings(num_classes=num_classes, in_channels=in_channels, layers=[2, 2, 2, 2], site_number=site_number, embed_dim=embed_dim, use_hypnns=True, version=2, lightweight=True, layer_number=layer_number)
        elif model_name == 'resnet18affine1':
            model = ResNetWithEmbeddings(num_classes=num_classes, in_channels=in_channels, layers=[2, 2, 2, 2], site_number=site_number, embed_dim=embed_dim, use_hypnns=True, version=1, lightweight=True, affine=True, layer_number=layer_number)
        elif model_name == 'resnet18affine2':
            model = ResNetWithEmbeddings(num_classes=num_classes, in_channels=in_channels, layers=[2, 2, 2, 2], site_number=site_number, embed_dim=embed_dim, use_hypnns=True, version=2, lightweight=True, affine=True, layer_number=layer_number)
        elif model_name == 'resnet18medium1':
            model = ResNetWithEmbeddings(num_classes=num_classes, in_channels=in_channels, layers=[2, 2, 2, 2], site_number=site_number, embed_dim=embed_dim, use_hypnns=True, version=1, lightweight=True, affine=True, medium_ffwrd=True, layer_number=layer_number)
        elif model_name == 'resnet18medium2':
            model = ResNetWithEmbeddings(num_classes=num_classes, in_channels=in_channels, layers=[2, 2, 2, 2], site_number=site_number, embed_dim=embed_dim, use_hypnns=True, version=2, lightweight=True, affine=True, medium_ffwrd=True, layer_number=layer_number)
        elif model_name == 'resnet34':
            model = ResNet34Model(num_classes=num_classes, in_channels=in_channels, pretrained=pretrained)
        elif model_name == 'resnet18':
            model = ResNet18Model(num_classes=num_classes, in_channels=in_channels, pretrained=pretrained)
        elif model_name == 'maxvitembv1':
            model = MaxViTEmb(num_classes=num_classes, in_channels=in_channels, depths=(2, 2, 2), channels=(64, 128, 256), site_number=site_number, latent_dim=embed_dim)
        elif model_name == 'maxvitv1':
            model = MaxViT(num_classes=num_classes, in_channels=in_channels, depths=(2, 2, 2), channels=(64, 128, 256))
        elif model_name == 'maxvitembv2':
            model = MaxViTEmb(num_classes=num_classes, in_channels=in_channels, depths=(2, 2, 2), channels=(32, 64, 128), site_number=site_number, latent_dim=embed_dim)
        elif model_name == 'maxvitv2':
            model = MaxViT(num_classes=num_classes, in_channels=in_channels, depths=(2, 2, 2), channels=(32, 64, 128))
        models.append(model)
    return models, num_classes