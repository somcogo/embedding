import math

import torch
from torch import nn

from .edmcode import UNetBlock, FeedForward

class ResNetWithEmbeddings(nn.Module):
    def __init__(self, num_classes, in_channels=3, embed_dim=2, layers=[3, 4, 6, 3], site_number=1, use_hypnns=False, version=None, lightweight=False, affine=False, medium_ffwrd=False, extra_lightweight=False, layer_number=4, conv1_residual=True, fc_residual=True):
        super().__init__()
        self.layer_number = layer_number
        self.conv1_residual = conv1_residual
        self.fc_residual = fc_residual
        self.in_channels = in_channels
        self.num_classes = num_classes

        self.embedding = nn.Embedding(site_number, embedding_dim=embed_dim)

        # self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=64, kernel_size=7, stride=2, padding=3, bias=False)
        if conv1_residual:
            self.conv1_weight = nn.Parameter(torch.empty((64, in_channels, 7, 7)))
            self.conv1_bias = nn.Parameter(torch.empty(64))
            nn.init.kaiming_uniform_(self.conv1_weight, a=math.sqrt(5))
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.conv1_weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.conv1_bias, -bound, bound)
        self.conv1_affine = nn.Linear(embed_dim, (in_channels*7*7 + 1)*64)
        self.first_ffwrd = FeedForward(in_channels=embed_dim, hidden_layer=64, out_channels=(in_channels*7*7 + 1)*64, version=version)


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
        if fc_residual:
            self.fc_weight = nn.Parameter(torch.empty((num_classes, 2**(layer_number + 5))))
            self.fc_bias = nn.Parameter(torch.empty(num_classes))
            nn.init.kaiming_uniform_(self.fc_weight, a=math.sqrt(5))
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.fc_weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.fc_bias, -bound, bound)
        self.fc_affine = nn.Linear(embed_dim, num_classes*(1 + 2**(layer_number + 5)))
        self.last_ffwrd = FeedForward(in_channels=embed_dim, hidden_layer=64, out_channels=num_classes*(1 + 2**(layer_number + 5)), version=version)

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
        conv1_params = self.first_ffwrd(emb)
        conv1_weight = conv1_params[:self.in_channels*7*7*64]
        conv1_weight = conv1_weight.reshape(64, self.in_channels, 7, 7)
        conv1_bias = conv1_params[self.in_channels*7*7*64:]
        if self.conv1_residual:
            conv1_weight = conv1_weight + self.conv1_weight
            conv1_bias = conv1_bias + self.conv1_bias
        x = nn.functional.conv2d(x, conv1_weight, conv1_bias)

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
        fc_params = self.last_ffwrd(emb)
        fc_weight = fc_params[:self.num_classes*2**(5 + self.layer_number)]
        fc_weight = fc_weight.reshape(self.num_classes, 2**(5 + self.layer_number))
        fc_bias = fc_params[self.num_classes*2**(5 + self.layer_number):]
        if self.fc_residual:
            fc_weight = fc_weight + self.fc_weight
            fc_bias = fc_bias + self.fc_bias
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