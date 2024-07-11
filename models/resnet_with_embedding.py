import math

import torch
from torch import nn

from .embedding_functionals import MODE_NAMES, GeneralConv2d, GeneralConvTranspose2d, GeneralLinear, GeneralBatchNorm2d, WeightGenerator

### ----- Based on the official PyTorch ResNet-18 implementation ----- ###
class CustomResnet(nn.Module):
    def __init__(self, channels=3, layers=[2, 2, 2, 2], feature_dims=[64, 128, 256, 512], norm_layer=GeneralBatchNorm2d, cifar=False, comb_gen_length=0, **kwargs):
        super().__init__()
        self.comb_gen_length = comb_gen_length
        self.kwargs = kwargs

        if cifar:
            self.conv1 = GeneralConv2d(channels, feature_dims[0], kernel_size=3, stride=1, padding=1, bias=False, **kwargs)
        else:
            self.conv1 = GeneralConv2d(channels, feature_dims[0], kernel_size=7, stride=2, padding=3, bias=False, **kwargs)
        self.batch_norm1 = norm_layer(num_features=feature_dims[0], **kwargs)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layers = nn.ModuleList([])
        self.make_layer(feature_dims[0], feature_dims[0], layers[0], norm_layer=norm_layer, **kwargs)
        self.make_layer(feature_dims[0], feature_dims[1], layers[1], stride=2, norm_layer=norm_layer, **kwargs)
        self.make_layer(feature_dims[1], feature_dims[2], layers[2], stride=2, norm_layer=norm_layer, **kwargs)
        self.make_layer(feature_dims[2], feature_dims[3], layers[3], stride=2, norm_layer=norm_layer, **kwargs)


    def make_layer(self, in_channels, out_channels, depth, stride=1, norm_layer=None, **kwargs):
        downsample = None
        if stride != 1 or in_channels != out_channels:
            downsample = nn.ModuleList([
                GeneralConv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=stride, **kwargs),
                norm_layer(out_channels, **kwargs)
            ])
        blocks = nn.ModuleList([])
        blocks.append(ResnetBlock(in_channels, out_channels, stride, downsample, norm_layer, **kwargs))

        for _ in range(1, depth):
            blocks.append(ResnetBlock(out_channels, out_channels, norm_layer=norm_layer, **kwargs))

        self.layers.append(blocks)

    def init_comb_gen_layers(self):
        comb_gen_layers = nn.ModuleList([])
        for i in range(self.comb_gen_length):
            if i == 0:
                comb_gen_layers.append(nn.Linear(self.kwargs['emb_dim'], 16))
            else:
                comb_gen_layers.append(nn.Linear(16, 16))
            comb_gen_layers.append(nn.ReLU())
        self.comb_gen_layers = comb_gen_layers
    
    def forward(self, x, emb):
        for layer in self.comb_gen_layers:
            emb = layer(emb)

        x = self.conv1(x, emb)
        x = self.batch_norm1(x, emb)
        x = self.relu(x)
        x = self.maxpool(x)

        features = []
        for layer in self.layers:
            for block in layer:
                x = block(x, emb)
            features.append(x)

        return features, emb

class ResnetBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None, norm_layer=GeneralBatchNorm2d, mode=None, **kwargs):
        super().__init__()
        self.residual_affine_generator = WeightGenerator(out_channels=out_channels, **kwargs) if mode in [MODE_NAMES['embedding'], MODE_NAMES['residual']] else None
        self.residual_const_generator = WeightGenerator(out_channels=out_channels, **kwargs) if mode in [MODE_NAMES['embedding'], MODE_NAMES['residual']] else None

        self.conv1 = GeneralConv2d(mode=mode, in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=stride, padding=1, **kwargs)
        self.batch_norm1 = norm_layer(mode=mode, num_features=out_channels, **kwargs)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = GeneralConv2d(mode=mode, in_channels=out_channels, out_channels=out_channels, kernel_size=3, padding=1, **kwargs)
        self.batch_norm2 = norm_layer(mode=mode, num_features=out_channels, **kwargs)
        self.downsample = downsample

    def forward(self, x, emb):
        out = self.conv1(x, emb)
        out = self.batch_norm1(out, emb)
        out = self.relu(out)

        if self.residual_affine_generator is not None:
            scale = self.residual_affine_generator(emb).unsqueeze(1).unsqueeze(1)
            const = self.residual_const_generator(emb).unsqueeze(1).unsqueeze(1)
            out = scale*out + const

        out = self.conv2(out, emb)
        out = self.batch_norm2(out, emb)

        if self.downsample is not None:
            x = self.downsample[0](x, emb)
            x = self.downsample[1](x, emb)

        out += x
        out = self.relu(out)

        return out