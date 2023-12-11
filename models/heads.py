import torch
import torch.nn as nn
import torch.nn.functional as F

from models.embedding_functionals import (GeneralAdaptiveAvgPool2d, GeneralBatchNorm2d,
                                          GeneralConv2d, GeneralLinear, GeneralReLU)
    
class ClassifierHead(nn.Module):
    def __init__(self, num_classes, mode='vanilla', weight_gen_args=None, **kwargs):
        super().__init__()

        self.avgpool = GeneralAdaptiveAvgPool2d((1, 1))
        self.fc = GeneralLinear(mode, 512, num_classes, **weight_gen_args)

    def forward(self, features, emb):
        x = features[-1]
        x = self.avgpool(x, emb)
        x = torch.flatten(x, 1)
        x = self.fc(x, emb)

        return x

### --------- TODO: cite https://github.com/yassouali/pytorch-segmentation/blob/master/models/upernet.py --------- ###
class PSPModule(nn.Module):
    # In the original inmplementation they use precise RoI pooling 
    # Instead of using adaptative average pooling
    def __init__(self, in_channels, mode, weight_gen_args, bin_sizes=[1, 2, 4, 6]):
        super().__init__()
        out_channels = in_channels // len(bin_sizes)
        self.stages = nn.ModuleList([self._make_stages(in_channels, out_channels, b_s, mode, weight_gen_args) 
                                                        for b_s in bin_sizes])
        self.conv = GeneralConv2d(mode=mode, in_channels=in_channels+(out_channels * len(bin_sizes)),
                                  out_channels=in_channels, kernel_size=3, padding=1, bias=False,
                                  **weight_gen_args),
        self.norm = GeneralBatchNorm2d(mode=mode, num_features=in_channels, **weight_gen_args),
        self.relu = nn.ReLU(inplace=True),
        self.dropout = nn.Dropout2d(0.1)

    def _make_stages(self, in_channels, out_channels, bin_sz, mode, weight_gen_args):
        prior = GeneralAdaptiveAvgPool2d(output_size=bin_sz)
        conv = GeneralConv2d(mode, in_channels, out_channels, kernel_size=1, bias=False, **weight_gen_args)
        bn = GeneralBatchNorm2d(mode, out_channels, **weight_gen_args)
        relu = GeneralReLU(inplace=True)
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
    def __init__(self, mode, weight_gen_args, feature_channels=[256, 512, 1024, 2048], fpn_out=256):
        super().__init__()
        assert feature_channels[0] == fpn_out
        self.conv1x1 = nn.ModuleList([GeneralConv2d(mode, ft_size, fpn_out, kernel_size=1, **weight_gen_args)
                                    for ft_size in feature_channels[1:]])
        self.smooth_conv =  nn.ModuleList([GeneralConv2d(mode, fpn_out, fpn_out, kernel_size=3, padding=1, **weight_gen_args)] 
                                    * (len(feature_channels)-1))
        self.conv_fusion = nn.ModuleList(
            GeneralConv2d(mode, len(feature_channels)*fpn_out, fpn_out, kernel_size=3, padding=1, bias=False, **weight_gen_args),
            GeneralBatchNorm2d(mode, fpn_out, **weight_gen_args),
            GeneralReLU(inplace=True)
        )

    def forward(self, features, emb):
        
        features[1:] = [conv1x1(feature, emb) for feature, conv1x1 in zip(features[1:], self.conv1x1)]
        P = [up_and_add(features[i], features[i-1]) for i in reversed(range(1, len(features)))]
        P = [smooth_conv(x, emb) for smooth_conv, x in zip(self.smooth_conv, P)]
        P = list(reversed(P))
        P.append(features[-1]) #P = [P1, P2, P3, P4]
        H, W = P[0].size(2), P[0].size(3)
        P[1:] = [F.interpolate(feature, size=(H, W), mode='bilinear', align_corners=True) for feature in P[1:]]

        x = torch.cat((P), dim=1)
        for module in self.conv_fusion:
            x = module(x)
        return x

class UperNet(nn.Module):
    # Implementing only the object path
    def __init__(self, mode, weight_gen_args, num_classes, fpn_out=256, feature_channels=[64, 128, 256, 512], input_size=(64, 64), **kwargs):
        super().__init__()

        self.input_size = input_size
        self.PPN = PSPModule(feature_channels[-1], mode, weight_gen_args)
        self.FPN = FPN_fuse(mode, weight_gen_args, feature_channels, fpn_out=fpn_out)
        self.head = GeneralConv2d(mode, fpn_out, num_classes, kernel_size=3, padding=1, **weight_gen_args)

    def forward(self, features, emb):        
        features[-1] = self.PPN(features[-1], emb)
        x = self.head(self.FPN(features, emb), emb)

        x = F.interpolate(x, size=self.input_size, mode='bilinear')
        return x

def get_head(head_name, **model_init):
    if head_name == 'classifier':
        head = ClassifierHead(**model_init)
    elif head_name == 'upernet':
        head = UperNet(**model_init)

    return head