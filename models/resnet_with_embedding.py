import math

import torch
from torch import nn

from .edmcode import UNetBlock, FeedForward
from .embedding_functionals import MODE_NAMES, GeneralConv2d, GeneralConvTranspose2d, GeneralLinear, GeneralBatchNorm2d, WeightGenerator

weight_gen_args = {'emb_dim':4,
                   'size': 1,
                   'gen_depth': 2,
                   'gen_affine': False,
                   'hidden_layer': 64}

### ----- Based on the official PyTorch ResNet-18 implementation ----- ###
class CustomResnet(nn.Module):
    def __init__(self, num_classes, in_channels=3, layers=[2, 2, 2, 2], mode='vanilla', weight_gen_args=None, norm_layer=GeneralBatchNorm2d, device=None, cifar=False, extra_conv=False):
        super().__init__()
        self.mode = mode

        self.embedding = nn.Parameter(torch.zeros(weight_gen_args['emb_dim'])) if mode in [MODE_NAMES['embedding'], MODE_NAMES['residual']] else None

        if extra_conv:
            self.conv0 = nn.Conv2d(in_channels, in_channels, kernel_size=1, bias=True)
            nn.init.eye_(self.conv0.weight[:,:,0,0])
        if cifar:
            self.conv1 = GeneralConv2d(mode, in_channels, 64, kernel_size=3, stride=1, padding=1, bias=False, weight_gen_args=weight_gen_args)
        else:
            self.conv1 = GeneralConv2d(mode, in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False, weight_gen_args=weight_gen_args)
        self.batch_norm1 = norm_layer(mode, 64, weight_gen_args=weight_gen_args)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.resnet_blocks = nn.ModuleList([])
        self.make_layer(64, 64, layers[0], norm_layer=norm_layer, weight_gen_args=weight_gen_args)
        self.make_layer(64, 128, layers[1], stride=2, norm_layer=norm_layer, weight_gen_args=weight_gen_args)
        self.make_layer(128, 256, layers[2], stride=2, norm_layer=norm_layer, weight_gen_args=weight_gen_args)
        self.make_layer(256, 512, layers[3], stride=2, norm_layer=norm_layer, weight_gen_args=weight_gen_args)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = GeneralLinear(mode, 512, num_classes, weight_gen_args=weight_gen_args)

    def make_layer(self, in_channels, out_channels, depth, stride=1, norm_layer=None, weight_gen_args=None):
        downsample = None
        if stride != 1 or in_channels != out_channels:
            downsample = nn.ModuleList([
                GeneralConv2d(self.mode, in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=stride, weight_gen_args=weight_gen_args),
                norm_layer(self.mode, out_channels, weight_gen_args=weight_gen_args)
            ])
        
        self.resnet_blocks.append(ResnetBlock(self.mode, in_channels, out_channels, stride, downsample, norm_layer, weight_gen_args))

        for _ in range(1, depth):
            self.resnet_blocks.append(ResnetBlock(self.mode, out_channels, out_channels, norm_layer=norm_layer, weight_gen_args=weight_gen_args))
    
    def forward(self, x):
        if hasattr(self, 'conv0'):
            x = self.conv0(x)
        x = self.conv1(x, self.embedding)
        x = self.batch_norm1(x, self.embedding)
        x = self.relu(x)
        x = self.maxpool(x)

        for block in self.resnet_blocks:
            x = block(x, self.embedding)
        x = self.fc(self.avgpool(x).reshape(x.shape[0], 512), self.embedding)
        return x

class ResnetBlock(nn.Module):
    def __init__(self, mode, in_channels, out_channels, stride=1, downsample=None, norm_layer=GeneralBatchNorm2d, weight_gen_args:dict =None):
        super().__init__()
        self.residual_affine_generator = WeightGenerator(weight_gen_args, out_channels) if mode is not MODE_NAMES['vanilla'] else None
        self.residual_const_generator = WeightGenerator(weight_gen_args, out_channels) if mode is not MODE_NAMES['vanilla'] else None

        self.conv1 = GeneralConv2d(mode=mode, in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=stride, padding=1, weight_gen_args=weight_gen_args)
        self.batch_norm1 = norm_layer(mode, out_channels, weight_gen_args=weight_gen_args)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = GeneralConv2d(mode=mode, in_channels=out_channels, out_channels=out_channels, kernel_size=3, padding=1, weight_gen_args=weight_gen_args)
        self.batch_norm2 = norm_layer(mode, out_channels, weight_gen_args=weight_gen_args)
        self.downsample = downsample

    def forward(self, x, emb):
        identity = x

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
            identity = self.downsample[1](x, emb)

        out += identity
        out = self.relu(out)

        return out
    
def get_backbone(model_name, **model_kwargs):
    if model_name == 'resnet18':
        backbone = CustomResnet(**model_kwargs)

    return backbone

class ResNetWithEmbeddings(nn.Module):
    def __init__(self, num_classes, in_channels=3, embed_dim=2, layers=[3, 4, 6, 3], site_number=1, use_hypnns=False, version=None, lightweight=False, affine=False, medium_ffwrd=False, extra_lightweight=False, layer_number=4, conv1_residual=True, fc_residual=True, cifar=False):
        super().__init__()
        self.layer_number = layer_number
        self.use_hypnns = use_hypnns
        self.conv1_residual = conv1_residual
        self.fc_residual = fc_residual
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.cifar = cifar

        self.embedding = nn.Embedding(site_number, embedding_dim=embed_dim)

        # self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=64, kernel_size=7, stride=2, padding=3, bias=False)
        if cifar:
            if use_hypnns:
                self.first_ffwrd = FeedForward(in_channels=embed_dim, hidden_layer=64, out_channels=(in_channels*3*3 + 1)*64, version=version)
            if (use_hypnns and conv1_residual) or not use_hypnns:
                self.conv1_weight = nn.Parameter(torch.empty((64, in_channels, 3, 3)))
                self.conv1_bias = nn.Parameter(torch.empty(64))
                nn.init.kaiming_uniform_(self.conv1_weight, a=math.sqrt(5))
                fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.conv1_weight)
                bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
                nn.init.uniform_(self.conv1_bias, -bound, bound)
        else:
            if use_hypnns:
                self.first_ffwrd = FeedForward(in_channels=embed_dim, hidden_layer=64, out_channels=(in_channels*7*7 + 1)*64, version=version)
            if (use_hypnns and conv1_residual) or not use_hypnns:
                self.conv1_weight = nn.Parameter(torch.empty((64, in_channels, 7, 7)))
                self.conv1_bias = nn.Parameter(torch.empty(64))
                nn.init.kaiming_uniform_(self.conv1_weight, a=math.sqrt(5))
                fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.conv1_weight)
                bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
                nn.init.uniform_(self.conv1_bias, -bound, bound)
        # self.conv1_affine = nn.Linear(embed_dim, (in_channels*7*7 + 1)*64)


        self.norm1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.ffwrd0 = FeedForward(in_channels=embed_dim, hidden_layer=64, out_channels=9, version=version) if extra_lightweight else None
        self.ffwrd1 = FeedForward(in_channels=embed_dim, hidden_layer=64, out_channels=9, version=version) if extra_lightweight else None
        self.ffwrd2 = FeedForward(in_channels=embed_dim, hidden_layer=64, out_channels=9, version=version) if extra_lightweight else None
        self.ffwrd3 = FeedForward(in_channels=embed_dim, hidden_layer=64, out_channels=9, version=version) if extra_lightweight else None

        self.ffwrd_a0 = FeedForward(in_channels=embed_dim, hidden_layer=64, out_channels=9*9, version=version) if extra_lightweight else None
        self.ffwrd_a1 = FeedForward(in_channels=embed_dim, hidden_layer=64, out_channels=9*9, version=version) if extra_lightweight else None
        self.ffwrd_a2 = FeedForward(in_channels=embed_dim, hidden_layer=64, out_channels=9*9, version=version) if extra_lightweight else None
        self.ffwrd_a3 = FeedForward(in_channels=embed_dim, hidden_layer=64, out_channels=9*9, version=version) if extra_lightweight else None

        self.layer0 = self._make_layer(layers[0], in_channels=64, out_channels=64,  embed_dim=embed_dim, use_hypnns=use_hypnns, version=version, lightweight=lightweight, ffwrd=self.ffwrd0, affine=affine, ffwrd_a=self.ffwrd_a0, medium_ffwrd=medium_ffwrd)

        self.layer1 = self._make_layer(layers[1], in_channels=64, out_channels=128, embed_dim=embed_dim, use_hypnns=use_hypnns, version=version, lightweight=lightweight, ffwrd=self.ffwrd1, affine=affine, ffwrd_a=self.ffwrd_a1, medium_ffwrd=medium_ffwrd)
    
        if layer_number > 2:
            self.layer2 = self._make_layer(layers[2], in_channels=128, out_channels=256, embed_dim=embed_dim, use_hypnns=use_hypnns, version=version, lightweight=lightweight, ffwrd=self.ffwrd2, affine=affine, ffwrd_a=self.ffwrd_a2, medium_ffwrd=medium_ffwrd)

        if layer_number > 3:
            self.layer3 = self._make_layer(layers[3], in_channels=256, out_channels=512, embed_dim=embed_dim, use_hypnns=use_hypnns, version=version, lightweight=lightweight, ffwrd=self.ffwrd3, affine=affine, ffwrd_a=self.ffwrd_a3, medium_ffwrd=medium_ffwrd)

        self.avgpool = nn.AdaptiveAvgPool2d((1,1))

        # self.fc = nn.Linear(2**(layer_number + 5), num_classes)
        if use_hypnns:
            self.last_ffwrd = FeedForward(in_channels=embed_dim, hidden_layer=64, out_channels=num_classes*(1 + 2**(layer_number + 5)), version=version)
        if (use_hypnns and fc_residual) or not use_hypnns:
            self.fc_weight = nn.Parameter(torch.empty((num_classes, 2**(layer_number + 5))))
            self.fc_bias = nn.Parameter(torch.empty(num_classes))
            nn.init.kaiming_uniform_(self.fc_weight, a=math.sqrt(5))
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.fc_weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.fc_bias, -bound, bound)
        # self.fc_affine = nn.Linear(embed_dim, num_classes*(1 + 2**(layer_number + 5)))

    def _make_layer(self, depth, in_channels, out_channels, embed_dim, use_hypnns=False, version=None, lightweight=None, ffwrd=None, affine=None, ffwrd_a=None, medium_ffwrd=False):
        
        blocks = nn.ModuleList()
        blocks.append(UNetBlock(in_channels=in_channels, out_channels=out_channels, emb_channels=embed_dim, use_hypnns=use_hypnns, version=version, ffwrd=ffwrd, lightweight=lightweight, affine_mode=affine, ffwrd_a=ffwrd_a, medium_ffwrd=medium_ffwrd))

        for _ in range(1, depth):
            blocks.append(UNetBlock(in_channels=out_channels, out_channels=out_channels, emb_channels=embed_dim, use_hypnns=use_hypnns, version=version, ffwrd=ffwrd, lightweight=lightweight, affine_mode=affine, ffwrd_a=ffwrd_a, medium_ffwrd=medium_ffwrd))

        return blocks

    def forward(self, x, site_id):

        emb = self.embedding(site_id).float()
        # if len(emb.shape) == 1:
        #     emb = emb.repeat(x.shape[0]).view(x.shape[0], -1)
        # if len(emb.shape) == 1:
        #     emb = emb.unsqueeze(0)

        # conv1_params = self.conv1_affine(emb)
        if self.cifar:
            if self.use_hypnns:
                conv1_params = self.first_ffwrd(emb)
                conv1_weight = conv1_params[:self.in_channels*3*3*64]
                conv1_weight = conv1_weight.reshape(64, self.in_channels, 3, 3)
                conv1_bias = conv1_params[self.in_channels*3*3*64:]
                if self.conv1_residual:
                    conv1_weight = conv1_weight + self.conv1_weight
                    conv1_bias = conv1_bias + self.conv1_bias
            else:
                conv1_weight = self.conv1_weight
                conv1_bias = self.conv1_bias
            x = nn.functional.conv2d(x, conv1_weight, conv1_bias, stride=1, padding=3)
        else:
            if self.use_hypnns:
                conv1_params = self.first_ffwrd(emb)
                conv1_weight = conv1_params[:self.in_channels*7*7*64]
                conv1_weight = conv1_weight.reshape(64, self.in_channels, 7, 7)
                conv1_bias = conv1_params[self.in_channels*7*7*64:]
                if self.conv1_residual:
                    conv1_weight = conv1_weight + self.conv1_weight
                    conv1_bias = conv1_bias + self.conv1_bias
            else:
                conv1_weight = self.conv1_weight
                conv1_bias = self.conv1_bias
            x = nn.functional.conv2d(x, conv1_weight, conv1_bias, stride=2, padding=3)

        x = self.pool(self.relu(self.norm1(x)))
        # emb = emb.repeat(x.shape[0]).view(x.shape[0], -1)
        for block in self.layer0:
            x = block(x, emb)
        for block in self.layer1:
            x = block(x, emb)
        if self.layer_number > 2:
            for block in self.layer2:
                x = block(x, emb)
        if self.layer_number > 3:
            for block in self.layer3:
                x = block(x, emb)
        x = self.avgpool(x).reshape(-1, 32*(2**self.layer_number))

        # fc_params = self.fc_affine(emb)
        if self.use_hypnns:
            fc_params = self.last_ffwrd(emb)
            fc_weight = fc_params[:self.num_classes*2**(5 + self.layer_number)]
            fc_weight = fc_weight.reshape(self.num_classes, 2**(5 + self.layer_number))
            fc_bias = fc_params[self.num_classes*2**(5 + self.layer_number):]
            if self.fc_residual:
                fc_weight = fc_weight + self.fc_weight
                fc_bias = fc_bias + self.fc_bias
        else:
            fc_weight = self.fc_weight
            fc_bias = self.fc_bias
        out = nn.functional.linear(x, fc_weight, fc_bias)
        return out
    
class UNetWithEmbedding(nn.Module):
    def __init__(self, num_classes, in_channels, embed_dim=2, encoder_layers=[2, 2, 2, 2], decoder_layers=[2, 2, 2, 2], site_number=1):
        super().__init__()

        self.enc = nn.ModuleList()
        self.dec = nn.ModuleList()

        for level, depth in enumerate(encoder_layers):
            self.enc.append(self._make_layer(
                depth=depth,
                in_channels=in_channels if level == 0 else 32 * 2**level,
                out_channels=64 * 2**level,
                embed_dim=embed_dim,
                down=True if level == len(encoder_layers) - 1 else False
            ))

        for level, depth in enumerate(decoder_layers):
            self.dec.append(self._make_layer(
                depth=depth,
            ))

    def _make_layer(self, depth, in_channels, out_channels, embed_dim, up=False, down=False):
        blocks = nn.ModuleList()

        for ndx in range(depth):
            blocks.append(UNetBlock(
                in_channels=in_channels if ndx == 0 else out_channels,
                out_channels=out_channels,
                emb_channels=embed_dim,
                up=up if ndx == depth - 1 else False,
                down=down if ndx == depth - 1 else False 
            ))

        return blocks

    def forward(self, x, site_id):
        return x