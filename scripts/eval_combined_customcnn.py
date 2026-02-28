import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    roc_curve,
    precision_recall_curve,
)
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms


OUT_ROOT = Path("combined_isic_binary")
TRAIN_CSV = OUT_ROOT / "train.csv"
TEST_CSV = OUT_ROOT / "test.csv"
CKPT = OUT_ROOT / "custom_cnn_best.pth"

IMG_SIZE = 224
BATCH_SIZE = 32


class ManifestDataset(Dataset):
    def __init__(self, df, root: Path, transform=None):
        self.df = df.reset_index(drop=True)
        self.root = Path(root)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = self.root / row["filepath"]
        img = Image.open(img_path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        label = int(row["label"])
        return img, label


class CustomResNet(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        self.layer4 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.4),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.classifier(x)
        return x


def compute_mean_std(ds, batch_size=64):
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False)
    n_pixels = 0
    channel_sum = torch.zeros(3)
    channel_sum_sq = torch.zeros(3)
    for imgs, _ in loader:
        n_pixels += imgs.numel() / 3
        channel_sum += imgs.sum(dim=[0, 2, 3])
        channel_sum_sq += (imgs ** 2).sum(dim=[0, 2, 3])
    mean = (channel_sum / n_pixels).tolist()
    std = (channel_sum_sq / n_pixels - torch.tensor(mean) ** 2).sqrt().tolist()
    return mean, std


def main():
    import pandas as pd
    import matplotlib.pyplot as plt

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_df = pd.read_csv(TRAIN_CSV)
    test_df = pd.read_csv(TEST_CSV)

    base_transform = transforms.Compose(
        [transforms.Resize((IMG_SIZE, IMG_SIZE)), transforms.ToTensor()]
    )
    base_ds = ManifestDataset(train_df, OUT_ROOT, transform=base_transform)
    mean, std = compute_mean_std(base_ds)

    tfm = transforms.Compose(
        [
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )
    test_ds = ManifestDataset(test_df, OUT_ROOT, transform=tfm)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    model = CustomResNet(num_classes=2).to(device)
    model.load_state_dict(torch.load(CKPT, map_location=device))
    model.eval()

    probs_list, labels_list = [], []
    with torch.no_grad():
        for imgs, labels in test_loader:
            imgs = imgs.to(device)
            logits = model(imgs)
            probs = torch.softmax(logits, dim=1)[:, 1]
            probs_list.append(probs.cpu().numpy())
            labels_list.append(labels.numpy())

    probs = np.concatenate(probs_list)
    y_true = np.concatenate(labels_list)
    y_pred = (probs >= 0.5).astype(int)

    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    roc_auc = roc_auc_score(y_true, probs)
    pr_auc = average_precision_score(y_true, probs)

    fig_dir = OUT_ROOT / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(4, 4))
    plt.imshow(cm, cmap="Blues")
    plt.title("Confusion Matrix (Test)")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    for i in range(2):
        for j in range(2):
            plt.text(j, i, str(cm[i, j]), ha="center", va="center")
    plt.tight_layout()
    plt.savefig(fig_dir / "confusion_matrix.png", dpi=300)
    plt.close()

    fpr, tpr, _ = roc_curve(y_true, probs)
    plt.figure(figsize=(4, 4))
    plt.plot(fpr, tpr, label=f"AUC={roc_auc:.4f}")
    plt.plot([0, 1], [0, 1], "--", color="gray")
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.title("ROC Curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig(fig_dir / "roc_curve.png", dpi=300)
    plt.close()

    pr, rc, _ = precision_recall_curve(y_true, probs)
    plt.figure(figsize=(4, 4))
    plt.plot(rc, pr, label=f"AP={pr_auc:.4f}")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("PR Curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig(fig_dir / "pr_curve.png", dpi=300)
    plt.close()

    (OUT_ROOT / "preds_test.csv").write_text(
        "prob_malignant,label\n"
        + "\n".join([f"{p:.6f},{y}" for p, y in zip(probs, y_true)]),
        encoding="utf-8",
    )

    results = {
        "accuracy": float(acc),
        "precision": float(prec),
        "recall": float(rec),
        "f1": float(f1),
        "roc_auc": float(roc_auc),
        "pr_auc": float(pr_auc),
        "mean": mean,
        "std": std,
        "train_size": int(len(train_df)),
        "test_size": int(len(test_df)),
    }
    (OUT_ROOT / "results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")

    print("Saved results.json and figures.")


if __name__ == "__main__":
    main()
