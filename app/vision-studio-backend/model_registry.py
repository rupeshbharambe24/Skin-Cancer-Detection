from dataclasses import dataclass
from pathlib import Path
from typing import Callable, List

import torch
import torch.nn as nn
import timm
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
    test_accuracy: float


def build_resnet18(num_classes: int = 2) -> nn.Module:
    model = models.resnet18(weights=None)
    in_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(0.3),
        nn.Linear(in_features, num_classes),
    )
    return model


def build_densenet121(num_classes: int = 2) -> nn.Module:
    model = models.densenet121(weights=None)
    in_features = model.classifier.in_features
    model.classifier = nn.Sequential(
        nn.Dropout(0.3),
        nn.Linear(in_features, num_classes),
    )
    return model


def build_mobilenetv3(num_classes: int = 2) -> nn.Module:
    model = models.mobilenet_v3_large(weights=None)
    in_features = model.classifier[3].in_features
    model.classifier[3] = nn.Linear(in_features, num_classes)
    return model


def build_vit_small_timm(num_classes: int = 2) -> nn.Module:
    return timm.create_model(
        "vit_small_patch16_224.augreg_in21k_ft_in1k",
        pretrained=False,
        num_classes=num_classes,
    )


def build_convnext_tiny_timm(num_classes: int = 2) -> nn.Module:
    return timm.create_model(
        "convnext_tiny.in12k_ft_in1k",
        pretrained=False,
        num_classes=num_classes,
    )


def build_efficientnetv2_s_timm(num_classes: int = 2) -> nn.Module:
    return timm.create_model(
        "tf_efficientnetv2_s.in21k_ft_in1k",
        pretrained=False,
        num_classes=num_classes,
    )


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
            test_accuracy=83.48,
        ),
        ModelSpec(
            key="resnet18_v2",
            name="ResNet18 (v2)",
            builder=build_resnet18,
            weights_path=BASE_DIR / "models" / "skin_lesions_model_resnet18.pth",
            input_size=224,
            norm_mean=[0.485, 0.456, 0.406],
            norm_std=[0.229, 0.224, 0.225],
            class_names=["benign", "malignant"],
            test_accuracy=90.0,
        ),
        ModelSpec(
            key="resnet18_v3",
            name="ResNet18 (v3)",
            builder=build_resnet18,
            weights_path=BASE_DIR / "models" /"skin_lesions_model_resnet18_v3.pth",
            input_size=224,
            norm_mean=[0.485, 0.456, 0.406],
            norm_std=[0.229, 0.224, 0.225],
            class_names=["benign", "malignant"],
            test_accuracy=88.0,
        ),
        ModelSpec(
            key="densenet121",
            name="DenseNet121",
            builder=build_densenet121,
            weights_path=BASE_DIR / "models" / "skin_lesions_model_densenet121.pth",
            input_size=224,
            norm_mean=[0.485, 0.456, 0.406],
            norm_std=[0.229, 0.224, 0.225],
            class_names=["benign", "malignant"],
            test_accuracy=90.0,
        ),
        ModelSpec(
            key="mobilenetv3",
            name="MobileNetV3-Large",
            builder=build_mobilenetv3,
            weights_path=BASE_DIR / "models" / "skin_lesions_model_mobilenetv3.pth",
            input_size=224,
            norm_mean=[0.485, 0.456, 0.406],
            norm_std=[0.229, 0.224, 0.225],
            class_names=["benign", "malignant"],
            test_accuracy=83.0,
        ),
        ModelSpec(
            key="vit_small_patch16_224",
            name="ViT-Small/16 (IN21K->IN1K)",
            builder=build_vit_small_timm,
            weights_path=BASE_DIR / "models" / "best_vit_small_patch16_224_augreg_in21k_ft_in1k.pt",
            input_size=224,
            norm_mean=[0.485, 0.456, 0.406],
            norm_std=[0.229, 0.224, 0.225],
            class_names=["benign", "malignant"],
            test_accuracy=90.02,
        ),
        ModelSpec(
            key="convnext_tiny_in12k",
            name="ConvNeXt-Tiny (IN12K->IN1K)",
            builder=build_convnext_tiny_timm,
            weights_path=BASE_DIR / "best_convnext_tiny_in12k_ft_in1k.pt",
            input_size=224,
            norm_mean=[0.485, 0.456, 0.406],
            norm_std=[0.229, 0.224, 0.225],
            class_names=["benign", "malignant"],
            test_accuracy=89.0,
        ),
        ModelSpec(
            key="efficientnetv2_s_in21k",
            name="EfficientNetV2-S (IN21K->IN1K)",
            builder=build_efficientnetv2_s_timm,
            weights_path=BASE_DIR / "best_tf_efficientnetv2_s_in21k_ft_in1k.pt",
            input_size=224,
            norm_mean=[0.485, 0.456, 0.406],
            norm_std=[0.229, 0.224, 0.225],
            class_names=["benign", "malignant"],
            test_accuracy=89.0,
        ),
    ]


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())
