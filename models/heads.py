import torch
import torch.nn as nn
import torch.nn.functional as F

from models.embedding_functionals import GeneralAdaptiveAvgPool2d, GeneralLinear, GeneralBatchNorm1d
    
class ClassifierHead(nn.Module):
    def __init__(self, feature_dims, num_classes, mode, norm_layer, **kwargs):
        super().__init__()

        self.avgpool = GeneralAdaptiveAvgPool2d((1, 1))
        self.fc = GeneralLinear(in_channels=feature_dims[-1], out_channels=num_classes, **kwargs)
        self.norm = GeneralBatchNorm1d(num_features=feature_dims[-1], **kwargs)

    def forward(self, _,  features, emb):
        x = features[-1]

        x = self.avgpool(x, emb)
        x = torch.flatten(x, 1)
        x = self.norm(x, emb)
        x = self.fc(x, emb)
        return x