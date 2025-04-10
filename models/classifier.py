from torchvision import models
import torch.nn as nn

class PretrainedClassifier(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        # 加载预训练ResNet-18
        self.base_model = models.resnet18(pretrained=False)
        in_features = self.base_model.fc.in_features
        self.base_model.fc = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.base_model(x)