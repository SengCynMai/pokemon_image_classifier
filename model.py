import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights

def get_model(num_classes, pretrained=True, freeze=True):
    weights = ResNet18_Weights.DEFAULT if pretrained else None
    model = resnet18(weights=weights)

    if freeze:
        for param in model.parameters():
            param.requires_grad = False

    model.fc = nn.Linear(model.fc.in_features, num_classes)

    return model