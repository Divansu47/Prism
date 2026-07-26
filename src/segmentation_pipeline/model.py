import torch
import torch.nn as nn
from torchvision.models.segmentation import deeplabv3_resnet50


class FoodSegmentationModel(nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()
        self.model = deeplabv3_resnet50(weights=None)
        self.model.classifier[4] = nn.Conv2d(256, num_classes, kernel_size=1)

    def forward(self, x):
        return self.model(x)['out']
