from pathlib import Path
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision import datasets, models, transforms


def build_resnet18(num_classes: int = 2) -> nn.Module:
    model = models.resnet18(weights=None)
    in_features = model.fc.in_features
    model.fc = nn.Sequential(nn.Dropout(0.3), nn.Linear(in_features, num_classes))
    return model


@torch.no_grad()
def predict_all(model: nn.Module, ds: datasets.ImageFolder, device: torch.device) -> Tuple[np.ndarray, np.ndarray]:
    model.eval()
    loader = torch.utils.data.DataLoader(ds, batch_size=32, shuffle=False, num_workers=0)
    probs_all, y_all = [], []
    for x, y in loader:
        x = x.to(device)
        logits = model(x)
        probs_all.append(F.softmax(logits, dim=1).cpu().numpy())
        y_all.append(y.numpy())
    return np.concatenate(probs_all, 0), np.concatenate(y_all, 0)


def load_image(path: str, size: int = 224) -> np.ndarray:
    img = Image.open(path).convert("RGB").resize((size, size))
    return np.asarray(img)


def make_grid(cases: List[Tuple[str, str]], title: str, out_base: Path):
    # cases: (img_path, caption)
    n = len(cases)
    cols = 5
    rows = int(np.ceil(n / cols))
    plt.figure(figsize=(cols * 2.3, rows * 2.7))
    for i, (p, cap) in enumerate(cases):
        ax = plt.subplot(rows, cols, i + 1)
        ax.imshow(load_image(p))
        ax.set_title(cap, fontsize=8)
        ax.axis("off")
    plt.suptitle(title, y=1.02, fontsize=12)
    out_base.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_base.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.savefig(out_base.with_suffix(".pdf"), bbox_inches="tight")
    plt.close()


def main():
    root = Path("data") / "Skin Lesions Classification" / "test"
    weights = Path("models") / "skin_lesions_model_resnet18.pth"
    out_dir = Path("figures")
    out_dir.mkdir(exist_ok=True)

    tfm = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    ds = datasets.ImageFolder(str(root), transform=tfm)
    class_to_idx = ds.class_to_idx
    pos_idx = class_to_idx.get("malignant", 1)
    inv = {v: k for k, v in class_to_idx.items()}

    device = torch.device("cpu")
    model = build_resnet18(num_classes=len(ds.classes))
    model.load_state_dict(torch.load(weights, map_location="cpu"))
    model.to(device)

    probs, y_true = predict_all(model, ds, device)
    y_pred = probs.argmax(1)

    paths = [p for p, _ in ds.samples]
    wrong = np.where(y_pred != y_true)[0]
    if len(wrong) == 0:
        print("No misclassifications found.")
        return

    y_score = probs[:, pos_idx]

    # False positives: true benign but predicted malignant; high malignant prob.
    fp = [i for i in wrong if (y_true[i] != pos_idx and y_pred[i] == pos_idx)]
    fp = sorted(fp, key=lambda i: float(y_score[i]), reverse=True)[:5]

    # False negatives: true malignant but predicted benign; low malignant prob.
    fn = [i for i in wrong if (y_true[i] == pos_idx and y_pred[i] != pos_idx)]
    fn = sorted(fn, key=lambda i: float(y_score[i]))[:5]

    cases = []
    for i in fp + fn:
        t = inv[int(y_true[i])]
        p = inv[int(y_pred[i])]
        cases.append((paths[i], f"T:{t} P:{p}\nP(mal)={y_score[i]:.2f}"))

    make_grid(cases, "Binary Classification Failure Cases (ResNet18 v2, Test)", out_dir / "binary_resnet18_v2__failure_cases")
    print("Wrote failure cases figure.")


if __name__ == "__main__":
    main()

