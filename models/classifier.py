from torchvision import models
import torch.nn as nn

class PretrainedClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        # 加载预训练ResNet-18
        self.base_model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)

        # 修改输入层适应32x32输入（可选方案）
        self.base_model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.base_model.maxpool = nn.Identity()  # 移除原maxpool层

        # 修改最后的全连接层
        num_features = self.base_model.fc.in_features
        self.base_model.fc = nn.Sequential(
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.Linear(512, 10)
        )

    def forward(self, x):
        return self.base_model(x)