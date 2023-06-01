import torch.nn as  nn
from torchvision.models import resnet34, ResNet34_Weights, resnet18, ResNet18_Weights, swin_t

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
    
class ResNet18Model(nn.Module):
    def __init__(self, num_classes, pretrained=False):
        super().__init__()
        if pretrained:
            weights = ResNet18_Weights.DEFAULT
        else:
            weights = None
        self.model = resnet18(weights=weights)
        self.model.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        out = self.model(x)
        return out

class TinySwin(nn.Module):
    def __init__(self, num_classes, pretrained=False):
        super().__init__()

        if pretrained:
            weights = 'DEFAULT'
        else:
            weights = None
        self.tiny_swin = swin_t(weights=weights)
        self.tiny_swin.head = nn.Linear(in_features=768, out_features=num_classes)

    def forward(self, x):
        out = self.tiny_swin(x)
        return out
