import torch.nn as  nn
from torchvision.models import resnet34, ResNet34_Weights

from .resnet_with_embedding import ResNetWithEmbeddings
    
class ResNet34Model(nn.Module):
    def __init__(self, num_classes, pretrained=False):
        super().__init__()
        if pretrained:
            weights = ResNet34_Weights.DEFAULT
        else:
            weights = None
        self.model = resnet34(weights=weights)
        self.model.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        out = self.model(x)
        return out