# TODO: cite https://github.com/yassouali/pytorch-segmentation/blob/master/models/upernet.py

import torch
import torch.nn as nn

from models.embedding_functionals import MODE_NAMES
from models.resnet_with_embedding import get_backbone
from models.heads import get_head
    
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