from dataclasses import dataclass
from typing import Callable, List

import torch
import torch.nn as nn
from torchvision import models


class BasicBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, stride: int = 1) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.downsample = None
        if stride != 1 or in_ch != out_ch:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_ch),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.relu(out)
        return out


class CustomResNet(nn.Module):
    def __init__(self, num_classes: int = 2, dropout: float = 0.3) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.layer1 = nn.Sequential(
            BasicBlock(32, 32),
            BasicBlock(32, 32),
        )
        self.layer2 = nn.Sequential(
            BasicBlock(32, 64, stride=2),
            BasicBlock(64, 64),
        )
        self.layer3 = nn.Sequential(
            BasicBlock(64, 128, stride=2),
            BasicBlock(128, 128),
        )
        self.layer4 = nn.Sequential(
            BasicBlock(128, 256, stride=2),
            BasicBlock(256, 256),
        )
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(256, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.fc(x)
        return x


@dataclass(frozen=True)
class ModelSpec:
    key: str
    name: str
    builder: Callable[[], nn.Module]
    weights_path: str
    input_size: int
    norm_mean: List[float]
    norm_std: List[float]
    class_names: List[str]


def build_resnet18(num_classes: int = 2) -> nn.Module:
    model = models.resnet18(weights=None)
    in_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(0.3),
        nn.Linear(in_features, num_classes),
    )
    return model


def build_custom_resnet(num_classes: int = 2) -> nn.Module:
    return CustomResNet(num_classes=num_classes, dropout=0.3)


def get_model_specs() -> List[ModelSpec]:
    return [
        ModelSpec(
            key="custom_resnet",
            name="Custom ResNet",
            builder=build_custom_resnet,
            weights_path="skin_lesions_model.pth",
            input_size=224,
            norm_mean=[0.5, 0.5, 0.5],
            norm_std=[0.5, 0.5, 0.5],
            class_names=["benign", "malignant"],
        ),
        ModelSpec(
            key="resnet18_v2",
            name="ResNet18 (v2)",
            builder=build_resnet18,
            weights_path="skin_lesions_model_resnet18.pth",
            input_size=224,
            norm_mean=[0.485, 0.456, 0.406],
            norm_std=[0.229, 0.224, 0.225],
            class_names=["benign", "malignant"],
        ),
        ModelSpec(
            key="resnet18_v3",
            name="ResNet18 (v3)",
            builder=build_resnet18,
            weights_path="skin_lesions_model_resnet18_v3.pth",
            input_size=224,
            norm_mean=[0.485, 0.456, 0.406],
            norm_std=[0.229, 0.224, 0.225],
            class_names=["benign", "malignant"],
        ),
    ]


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())
