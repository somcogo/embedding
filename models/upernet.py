# TODO: cite https://github.com/yassouali/pytorch-segmentation/blob/master/models/upernet.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from utils.helpers import initialize_weights
from itertools import chain

from models.embedding_functionals import GeneralConv2d, GeneralBatchNorm2d, GeneralLinear

class PSPModule_emb(nn.Module):
    # In the original inmplementation they use precise RoI pooling 
    # Instead of using adaptative average pooling
    def __init__(self, in_channels, bin_sizes=[1, 2, 4, 6]):
        super(PSPModule, self).__init__()
        out_channels = in_channels // len(bin_sizes)
        self.stages = nn.ModuleList([self._make_stages(in_channels, out_channels, b_s) 
                                                        for b_s in bin_sizes])
        self.conv = GeneralConv2d(in_channels+(out_channels * len(bin_sizes)), in_channels, 
                                    kernel_size=3, padding=1, bias=False),
        self.norm = GeneralBatchNorm2d(in_channels),
        self.relu = nn.ReLU(inplace=True),
        self.dropout = nn.Dropout2d(0.1)

    def _make_stages(self, in_channels, out_channels, bin_sz):
        prior = nn.AdaptiveAvgPool2d(output_size=bin_sz)
        conv = GeneralConv2d(in_channels, out_channels, kernel_size=1, bias=False)
        bn = GeneralBatchNorm2d(out_channels)
        relu = nn.ReLU(inplace=True)
        return nn.ModuleList([prior, conv, bn, relu])
    
    def forward(self, features, emb):
        h, w = features.size()[2], features.size()[3]
        pyramids = [features]

        for stage in self.stages:
            f = features
            for module in stage:
                f = module(f, emb)
            pyramids.append(F.interpolate(f, size=(h, w), mode='bilinear', align_corners=True))
            
        fused = self.conv(torch.cat(pyramids, dim=1), emb)
        fused = self.norm(fused, emb)
        fused = self.relu(fused)
        output = self.dropout(fused)
        return output

def up_and_add(x, y):
    return F.interpolate(x, size=(y.size(2), y.size(3)), mode='bilinear', align_corners=True) + y

class FPN_fuse(nn.Module):
    def __init__(self, feature_channels=[256, 512, 1024, 2048], fpn_out=256):
        super(FPN_fuse, self).__init__()
        assert feature_channels[0] == fpn_out
        self.conv1x1 = nn.ModuleList([nn.Conv2d(ft_size, fpn_out, kernel_size=1)
                                    for ft_size in feature_channels[1:]])
        self.smooth_conv =  nn.ModuleList([nn.Conv2d(fpn_out, fpn_out, kernel_size=3, padding=1)] 
                                    * (len(feature_channels)-1))
        self.conv_fusion = nn.Sequential(
            nn.Conv2d(len(feature_channels)*fpn_out, fpn_out, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(fpn_out),
            nn.ReLU(inplace=True)
        )

    def forward(self, features):
        
        features[1:] = [conv1x1(feature) for feature, conv1x1 in zip(features[1:], self.conv1x1)]
        P = [up_and_add(features[i], features[i-1]) for i in reversed(range(1, len(features)))]
        P = [smooth_conv(x) for smooth_conv, x in zip(self.smooth_conv, P)]
        P = list(reversed(P))
        P.append(features[-1]) #P = [P1, P2, P3, P4]
        H, W = P[0].size(2), P[0].size(3)
        P[1:] = [F.interpolate(feature, size=(H, W), mode='bilinear', align_corners=True) for feature in P[1:]]

        x = self.conv_fusion(torch.cat((P), dim=1))
        return x

class UperNet(nn.Module):
    # Implementing only the object path
    def __init__(self, num_classes, in_channels=3, backbone='resnet101', pretrained=True, use_aux=True, fpn_out=256, freeze_bn=False, **_):
        super(UperNet, self).__init__()

        if backbone == 'resnet34' or backbone == 'resnet18':
            feature_channels = [64, 128, 256, 512]
        else:
            feature_channels = [256, 512, 1024, 2048]
        self.backbone = ResNet(in_channels, pretrained=pretrained)
        self.PPN = PSPModule(feature_channels[-1])
        self.FPN = FPN_fuse(feature_channels, fpn_out=fpn_out)
        self.head = nn.Conv2d(fpn_out, num_classes, kernel_size=3, padding=1)
        if freeze_bn: self.freeze_bn()

    def forward(self, x):
        input_size = (x.size()[2], x.size()[3])

        features = self.backbone(x)
        features[-1] = self.PPN(features[-1])
        x = self.head(self.FPN(features))

        x = F.interpolate(x, size=input_size, mode='bilinear')
        return x

    def get_backbone_params(self):
        return self.backbone.parameters()

    def get_decoder_params(self):
        return chain(self.PPN.parameters(), self.FPN.parameters(), self.head.parameters())

    def freeze_bn(self):
        for module in self.modules():
            if isinstance(module, nn.BatchNorm2d): module.eval()