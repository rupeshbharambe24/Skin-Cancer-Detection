from dataclasses import dataclass
from pathlib import Path
from typing import Callable, List

import torch
import torch.nn as nn
from torchvision import models

BASE_DIR = Path(__file__).resolve().parents[2]


class SkinLesionCNN(nn.Module):
    def __init__(self, num_classes: int = 2) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.5),
            nn.Linear(256 * 14 * 14, 512),
            nn.ReLU(),
            nn.Linear(512, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.classifier(x)
        return x


@dataclass(frozen=True)
class ModelSpec:
    key: str
    name: str
    builder: Callable[[], nn.Module]
    weights_path: Path
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


def build_custom_cnn(num_classes: int = 2) -> nn.Module:
    return SkinLesionCNN(num_classes=num_classes)


def get_model_specs() -> List[ModelSpec]:
    return [
        ModelSpec(
            key="custom_cnn_v1",
            name="Custom CNN (v1)",
            builder=build_custom_cnn,
            weights_path=BASE_DIR / "models" / "skin_lesions_model.pth",
            input_size=224,
            norm_mean=[0.5, 0.5, 0.5],
            norm_std=[0.5, 0.5, 0.5],
            class_names=["benign", "malignant"],
        ),
        ModelSpec(
            key="resnet18_v2",
            name="ResNet18 (v2)",
            builder=build_resnet18,
            weights_path=BASE_DIR / "models" /"skin_lesions_model_resnet18.pth",
            input_size=224,
            norm_mean=[0.485, 0.456, 0.406],
            norm_std=[0.229, 0.224, 0.225],
            class_names=["benign", "malignant"],
        ),
        ModelSpec(
            key="resnet18_v3",
            name="ResNet18 (v3)",
            builder=build_resnet18,
            weights_path=BASE_DIR / "models" / "skin_lesions_model_resnet18_v3.pth",
            input_size=224,
            norm_mean=[0.485, 0.456, 0.406],
            norm_std=[0.229, 0.224, 0.225],
            class_names=["benign", "malignant"],
        ),
    ]


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())
