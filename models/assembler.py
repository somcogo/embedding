# TODO: cite https://github.com/yassouali/pytorch-segmentation/blob/master/models/upernet.py

import torch
import torch.nn as nn

from models.embedding_functionals import MODE_NAMES
from models.resnet_with_embedding import CustomResnet
from models.convnext_emb import ConvNeXt
from models.convnext import ConvNeXt as ConvNeXtOG
from models.heads import ClassifierHead, UperNet
from models.swinv2 import SwinTransformerV2

def get_backbone(backbone_name, **model_config):
    if backbone_name == 'resnet':
        backbone = CustomResnet(**model_config)
    elif backbone_name == 'convnext':
        backbone = ConvNeXt(**model_config)
    elif backbone_name == 'convnextog':
        backbone = ConvNeXtOG(**model_config)
    elif backbone_name == 'swinv2':
        backbone = SwinTransformerV2(**model_config)

    return backbone

def get_head(head_name, **model_config):
    if head_name == 'classifier':
        head = ClassifierHead(**model_config)
    elif head_name == 'upernet':
        head = UperNet(**model_config)

    return head

class ModelAssembler(nn.Module):
    def __init__(self, mode='vanilla', emb_dim=None, **model_config):
        super().__init__()

        self.embedding = nn.Parameter(torch.zeros(emb_dim)) if mode in [MODE_NAMES['embedding'], MODE_NAMES['residual']] else None

        self.backbone = get_backbone(mode=mode, emb_dim=emb_dim, **model_config)
        self.head = get_head(mode=mode, emb_dim=emb_dim, **model_config)

    def forward(self, x):
        features = self.backbone(x, self.embedding)
        x = self.head(x, features, self.embedding)        
        return x