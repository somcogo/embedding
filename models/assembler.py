import torch
import torch.nn as nn

from models.embedding_functionals import MODE_NAMES, BatchNorm2d_emb_replace, InstanceNorm2d_emb_replace
from models.resnet_with_embedding import CustomResnet
from models.heads import ClassifierHead

def get_backbone(backbone_name, **model_config):
    if backbone_name == 'resnet':
        backbone = CustomResnet(**model_config)

    return backbone

def get_head(head_name, **model_config):
    if head_name == 'classifier':
        head = ClassifierHead(**model_config)

    return head

class ModelAssembler(nn.Module):
    def __init__(self, mode='vanilla', emb_dim=None, **model_config):
        super().__init__()

        self.embedding = nn.Parameter(torch.zeros(emb_dim, dtype=torch.float32)) if mode in [MODE_NAMES['embedding'], MODE_NAMES['residual'], MODE_NAMES['fedbn']] else None

        self.backbone = get_backbone(mode=mode, emb_dim=emb_dim, **model_config)
        self.head = get_head(mode=mode, emb_dim=emb_dim, **model_config)
        if type(self.backbone) == CustomResnet:
            self.backbone.init_comb_gen_layers()
        for m in self.backbone.modules():
            if type(m) in [BatchNorm2d_emb_replace, InstanceNorm2d_emb_replace]:
                m.init_norm_generator_params()
        for m in self.head.modules():
            if type(m) in [BatchNorm2d_emb_replace, InstanceNorm2d_emb_replace]:
                m.init_norm_generator_params()

    def forward(self, x):
        features, emb = self.backbone(x, self.embedding)
        x = self.head(x, features, emb)
        return x