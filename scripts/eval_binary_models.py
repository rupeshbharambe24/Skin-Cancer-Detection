import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    precision_recall_curve,
    precision_recall_fscore_support,
    roc_auc_score,
    roc_curve,
)
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms


FIG_DIR = Path("figures")
FIG_DIR.mkdir(exist_ok=True)


def _savefig(base: Path):
    base.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(base.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.savefig(base.with_suffix(".pdf"), bbox_inches="tight")
    plt.close()


def _count_images_by_class(root: Path) -> Dict[str, int]:
    out: Dict[str, int] = {}
    for cls_dir in sorted([p for p in root.iterdir() if p.is_dir()]):
        out[cls_dir.name] = len([p for p in cls_dir.rglob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png"}])
    return out


class SkinLesionCNN(nn.Module):
    # Matches `Skin_Lesions_train.ipynb` exactly.
    def __init__(self, num_classes: int = 2):
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
        return self.classifier(x)


def build_resnet18(num_classes: int = 2) -> nn.Module:
    # Matches `Skin_Lesions_train_v2.ipynb` and `Skin_Lesions_train_v3.ipynb`.
    model = models.resnet18(weights=None)
    in_features = model.fc.in_features
    model.fc = nn.Sequential(nn.Dropout(0.3), nn.Linear(in_features, num_classes))
    return model


@dataclass(frozen=True)
class BinaryModelSpec:
    key: str
    display_name: str
    weights_path: Path
    normalize: Tuple[Tuple[float, float, float], Tuple[float, float, float]]
    builder: str  # "cnn" or "resnet18"


def _infer_positive_class_idx(class_to_idx: Dict[str, int]) -> int:
    for k, v in class_to_idx.items():
        if "mal" in k.lower():
            return int(v)
    # Fallback: assume idx 1 is positive if present.
    return int(class_to_idx.get("malignant", 1))


@torch.no_grad()
def predict_probs(
    model: nn.Module, loader: DataLoader, device: torch.device
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    model.eval()
    probs_all: List[np.ndarray] = []
    y_all: List[np.ndarray] = []
    paths: List[str] = []
    for x, y in loader:
        x = x.to(device)
        logits = model(x)
        probs = F.softmax(logits, dim=1).cpu().numpy()
        probs_all.append(probs)
        y_all.append(y.numpy())
        if hasattr(loader.dataset, "samples"):
            # ImageFolder or Subset(ImageFolder)
            if isinstance(loader.dataset, torch.utils.data.Subset):
                samples = [loader.dataset.dataset.samples[i] for i in loader.dataset.indices]
            else:
                samples = loader.dataset.samples
            # Track paths approximately: batch-size aligned if no shuffle.
            # For evaluation we use shuffle=False, so indexing is safe.
            start = len(paths)
            end = start + len(y)
            paths.extend([s[0] for s in samples[start:end]])
    probs = np.concatenate(probs_all, axis=0)
    y_true = np.concatenate(y_all, axis=0)
    return probs, y_true, paths


def plot_confusion(cm: np.ndarray, class_names: List[str], title: str, out_base: Path):
    plt.figure(figsize=(5, 4))
    plt.imshow(cm, cmap="Blues")
    plt.title(title)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.xticks(range(len(class_names)), class_names, rotation=45, ha="right")
    plt.yticks(range(len(class_names)), class_names)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, str(cm[i, j]), ha="center", va="center", fontsize=10, color="black")
    _savefig(out_base)


def plot_roc(y_true_bin: np.ndarray, y_score: np.ndarray, title: str, out_base: Path):
    fpr, tpr, _ = roc_curve(y_true_bin, y_score)
    auc = roc_auc_score(y_true_bin, y_score)
    plt.figure(figsize=(5, 4))
    plt.plot(fpr, tpr, label=f"AUC={auc:.4f}")
    plt.plot([0, 1], [0, 1], "--", color="gray", linewidth=1)
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(title)
    plt.legend(loc="lower right")
    _savefig(out_base)
    return float(auc)


def plot_pr(y_true_bin: np.ndarray, y_score: np.ndarray, title: str, out_base: Path):
    p, r, _ = precision_recall_curve(y_true_bin, y_score)
    ap = average_precision_score(y_true_bin, y_score)
    plt.figure(figsize=(5, 4))
    plt.plot(r, p, label=f"AP={ap:.4f}")
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(title)
    plt.legend(loc="lower left")
    _savefig(out_base)
    return float(ap)


def evaluate_binary_models(
    data_root: Path = Path("data") / "Skin Lesions Classification",
    models_root: Path = Path("models"),
    device: str = "cpu",
    batch_size: int = 32,
) -> pd.DataFrame:
    device_t = torch.device(device)
    test_dir = data_root / "test"

    specs = [
        BinaryModelSpec(
            key="binary_custom_cnn_v1",
            display_name="Custom CNN (v1) [Skin_Lesions_train]",
            weights_path=models_root / "skin_lesions_model.pth",
            normalize=((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            builder="cnn",
        ),
        BinaryModelSpec(
            key="binary_resnet18_v2",
            display_name="ResNet18 (v2) [Skin_Lesions_train_v2]",
            weights_path=models_root / "skin_lesions_model_resnet18.pth",
            normalize=((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            builder="resnet18",
        ),
        BinaryModelSpec(
            key="binary_resnet18_v3",
            display_name="ResNet18 (v3) [Skin_Lesions_train_v3]",
            weights_path=models_root / "skin_lesions_model_resnet18_v3.pth",
            normalize=((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            builder="resnet18",
        ),
    ]

    rows: List[Dict[str, object]] = []

    # Dataset counts (for report)
    counts_train = _count_images_by_class(data_root / "train")
    counts_test = _count_images_by_class(data_root / "test")
    (FIG_DIR / "binary_dataset_counts.csv").write_text(
        pd.DataFrame({"split": ["train"] * len(counts_train) + ["test"] * len(counts_test),
                      "class": list(counts_train.keys()) + list(counts_test.keys()),
                      "count": list(counts_train.values()) + list(counts_test.values())}).to_csv(index=False),
        encoding="utf-8",
    )

    for spec in specs:
        if not spec.weights_path.exists():
            rows.append(
                {
                    "model_key": spec.key,
                    "model_name": spec.display_name,
                    "weights_path": str(spec.weights_path),
                    "status": "missing_weights",
                }
            )
            continue

        mean, std = spec.normalize
        tfm = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=list(mean), std=list(std)),
            ]
        )
        ds = datasets.ImageFolder(str(test_dir), transform=tfm)
        loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)

        if spec.builder == "cnn":
            model = SkinLesionCNN(num_classes=len(ds.classes))
        elif spec.builder == "resnet18":
            model = build_resnet18(num_classes=len(ds.classes))
        else:
            raise ValueError(spec.builder)

        state = torch.load(spec.weights_path, map_location="cpu")
        model.load_state_dict(state)
        model.to(device_t)

        probs, y_true, _paths = predict_probs(model, loader, device_t)
        y_pred = probs.argmax(axis=1)

        class_to_idx = ds.class_to_idx
        pos_idx = _infer_positive_class_idx(class_to_idx)
        y_true_bin = (y_true == pos_idx).astype(np.int32)
        y_score = probs[:, pos_idx]

        acc = float(accuracy_score(y_true, y_pred))
        prec, rec, f1, _ = precision_recall_fscore_support(
            y_true, y_pred, average="binary", pos_label=pos_idx, zero_division=0
        )
        roc_auc = float(roc_auc_score(y_true_bin, y_score))
        pr_auc = float(average_precision_score(y_true_bin, y_score))

        cm = confusion_matrix(y_true, y_pred, labels=list(range(len(ds.classes))))
        plot_confusion(
            cm,
            class_names=ds.classes,
            title=f"Confusion Matrix (Test) - {spec.display_name}",
            out_base=FIG_DIR / f"{spec.key}__confusion_matrix",
        )
        plot_roc(
            y_true_bin,
            y_score,
            title=f"ROC Curve (Test) - {spec.display_name}",
            out_base=FIG_DIR / f"{spec.key}__roc",
        )
        plot_pr(
            y_true_bin,
            y_score,
            title=f"Precision–Recall (Test) - {spec.display_name}",
            out_base=FIG_DIR / f"{spec.key}__pr",
        )

        rows.append(
            {
                "model_key": spec.key,
                "model_name": spec.display_name,
                "weights_path": str(spec.weights_path),
                "test_accuracy": acc,
                "test_precision": float(prec),
                "test_recall": float(rec),
                "test_f1": float(f1),
                "test_roc_auc": roc_auc,
                "test_pr_auc": pr_auc,
                "status": "ok",
            }
        )

    return pd.DataFrame(rows)


def main():
    plt.rcParams.update(
        {
            "figure.dpi": 120,
            "savefig.dpi": 300,
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "legend.fontsize": 9,
        }
    )
    df = evaluate_binary_models()
    out_csv = Path("binary_results.csv")
    df.to_csv(out_csv, index=False)
    print(f"Wrote: {out_csv}")
    print(df)


if __name__ == "__main__":
    main()

