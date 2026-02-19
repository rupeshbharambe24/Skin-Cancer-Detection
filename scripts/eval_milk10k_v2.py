import json
import math
import os
from pathlib import Path
from typing import Any, Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)


FIG_DIR = Path("figures")
FIG_DIR.mkdir(exist_ok=True)


def _savefig(base: Path):
    base.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(base.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.savefig(base.with_suffix(".pdf"), bbox_inches="tight")
    plt.close()


def load_notebook(nb_path: Path) -> Dict[str, Any]:
    return json.loads(nb_path.read_text(encoding="utf-8"))


def exec_cells(nb: Dict[str, Any], indices: List[int], g: Dict[str, Any]):
    for idx in indices:
        cell = nb["cells"][idx]
        if cell.get("cell_type") != "code":
            continue
        src = "".join(cell.get("source", []))
        # Skip pip install magics / shell escapes if present.
        if src.lstrip().startswith("!"):
            continue
        exec(compile(src, filename=f"<nb:{idx}>", mode="exec"), g, g)


@torch.no_grad()
def predict_on_loader(model, loader, device: torch.device) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    probs_sig_all: List[np.ndarray] = []
    logits_sm_all: List[np.ndarray] = []
    y_idx_all: List[np.ndarray] = []
    y_oh_all: List[np.ndarray] = []

    for batch in loader:
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
        batch["cat"] = {k: v.to(device) for k, v in batch["cat"].items()}

        out = model(batch)
        probs_sig = torch.sigmoid(out["logits_sig"]).cpu().numpy()
        logits_sm = out["logits_sm"].cpu().numpy()
        y_idx = batch["y_idx"].cpu().numpy()
        y_oh = batch["y_oh"].cpu().numpy()

        probs_sig_all.append(probs_sig)
        logits_sm_all.append(logits_sm)
        y_idx_all.append(y_idx)
        y_oh_all.append(y_oh)

    probs_sig = np.concatenate(probs_sig_all, axis=0)
    logits_sm = np.concatenate(logits_sm_all, axis=0)
    y_idx = np.concatenate(y_idx_all, axis=0)
    y_oh = np.concatenate(y_oh_all, axis=0)
    return probs_sig, logits_sm, y_idx, y_oh


def multilabel_metrics_at_threshold(
    probs: np.ndarray, y_true_idx: np.ndarray, n_classes: int, thr: float = 0.5
) -> Dict[str, Any]:
    y_true_oh = np.zeros((len(y_true_idx), n_classes), dtype=np.int32)
    y_true_oh[np.arange(len(y_true_idx)), y_true_idx.astype(int)] = 1
    y_pred_bin = (probs >= thr).astype(np.int32)

    per_class = []
    for c in range(n_classes):
        tp = int(np.sum((y_pred_bin[:, c] == 1) & (y_true_oh[:, c] == 1)))
        fp = int(np.sum((y_pred_bin[:, c] == 1) & (y_true_oh[:, c] == 0)))
        fn = int(np.sum((y_pred_bin[:, c] == 0) & (y_true_oh[:, c] == 1)))
        support = int(np.sum(y_true_oh[:, c] == 1))
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        denom = (2 * tp + fp + fn)
        f1 = (2 * tp / denom) if denom > 0 else 0.0
        per_class.append(
            {
                "class_idx": c,
                "support": support,
                "precision@0.5": float(prec),
                "recall@0.5": float(rec),
                "f1@0.5": float(f1),
                "tp": tp,
                "fp": fp,
                "fn": fn,
            }
        )

    macro_prec = float(np.mean([r["precision@0.5"] for r in per_class]))
    macro_rec = float(np.mean([r["recall@0.5"] for r in per_class]))
    macro_f1 = float(np.mean([r["f1@0.5"] for r in per_class]))
    return {"macro_precision@0.5": macro_prec, "macro_recall@0.5": macro_rec, "macro_f1@0.5": macro_f1, "per_class": per_class}


def top_confusions(cm: np.ndarray, class_names: List[str], k: int = 10) -> pd.DataFrame:
    rows = []
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            if i == j:
                continue
            rows.append((cm[i, j], class_names[i], class_names[j]))
    rows.sort(key=lambda x: x[0], reverse=True)
    rows = [r for r in rows if r[0] > 0][:k]
    return pd.DataFrame(rows, columns=["count", "true_class", "pred_class"])


def plot_confusion(cm: np.ndarray, class_names: List[str], out_base: Path):
    plt.figure(figsize=(9, 8))
    plt.imshow(cm, cmap="Blues")
    plt.title("Confusion Matrix (Argmax)")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.xticks(range(len(class_names)), class_names, rotation=45, ha="right")
    plt.yticks(range(len(class_names)), class_names)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            v = cm[i, j]
            if v == 0:
                continue
            plt.text(j, i, str(int(v)), ha="center", va="center", fontsize=8, color="black")
    _savefig(out_base)


def plot_per_class_f1(per_class_df: pd.DataFrame, class_names: List[str], out_base: Path):
    df = per_class_df.copy()
    df["class"] = [class_names[i] for i in df["class_idx"].tolist()]
    plt.figure(figsize=(10, 4))
    plt.bar(df["class"], df["f1@0.5"])
    plt.xticks(rotation=45, ha="right")
    plt.ylim(0, 1)
    plt.ylabel("F1 @ 0.5")
    plt.title("Per-class F1 @ 0.5 (Sigmoid Head)")
    _savefig(out_base)


def main():
    plt.rcParams.update(
        {
            "figure.dpi": 120,
            "savefig.dpi": 300,
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
        }
    )

    nb_path = Path("MILK10k_ViT_LatentFusion_v2_from_metadata.ipynb")
    ckpt_path = Path("best_milk10k_vit_latentfusion.pt")
    if not ckpt_path.exists():
        raise FileNotFoundError(str(ckpt_path))

    nb = load_notebook(nb_path)

    g: Dict[str, Any] = {}

    # Minimal Jupyter shims so notebook cells can run under plain Python.
    def display(*_args, **_kwargs):
        return None

    g["display"] = display

    # Execute the notebook cells needed to build val_loader and model definition.
    cell_indices = [4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32, 34]
    exec_cells(nb, cell_indices, g)

    model = g["model"]
    cfg = g["cfg"]
    val_loader = g["val_loader"]
    device = g["device"]

    # Load weights
    state = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(state, strict=True)
    model.to(device)

    # Inference
    probs_sig, logits_sm, y_true_idx, y_true_oh = predict_on_loader(model, val_loader, device)
    y_pred_idx = logits_sm.argmax(axis=1)

    class_names = list(cfg.classes)
    n_classes = len(class_names)

    # Metrics: argmax-based
    top1 = float(accuracy_score(y_true_idx, y_pred_idx))
    bacc = float(balanced_accuracy_score(y_true_idx, y_pred_idx))

    # Threshold-based macro metrics (sigmoid head) - computed two ways:
    # 1) custom TP/FP/FN bookkeeping
    ml = multilabel_metrics_at_threshold(probs_sig, y_true_idx, n_classes=n_classes, thr=0.5)
    # 2) notebook's dice-like macro-F1 helper (should match macro_f1@0.5)
    macro_f1_nb = float(g["macro_f1_threshold05"](probs_sig, y_true_oh))

    # Per-class table at threshold 0.5
    per_class_df = pd.DataFrame(ml["per_class"])
    per_class_df["class"] = [class_names[i] for i in per_class_df["class_idx"].tolist()]
    per_class_df = per_class_df[
        ["class_idx", "class", "support", "precision@0.5", "recall@0.5", "f1@0.5", "tp", "fp", "fn"]
    ].sort_values("class_idx")

    # Confusion matrix
    cm = confusion_matrix(y_true_idx, y_pred_idx, labels=list(range(n_classes)))
    cm_df = pd.DataFrame(cm, index=class_names, columns=class_names)

    # Classification report (argmax)
    report_text = classification_report(y_true_idx, y_pred_idx, target_names=class_names, zero_division=0)

    # Save artifacts
    overall = pd.DataFrame(
        [
            {
                "model_key": "milk10k_v2",
                "val_top1_accuracy": top1,
                "val_balanced_accuracy": bacc,
                "val_macro_precision@0.5": ml["macro_precision@0.5"],
                "val_macro_recall@0.5": ml["macro_recall@0.5"],
                "val_macro_f1@0.5": ml["macro_f1@0.5"],
                "val_macro_f1@0.5_notebook_fn": macro_f1_nb,
                "img_size": int(cfg.img_size),
                "val_split_ratio": 0.15,
                "split_seed": 42,
            }
        ]
    )
    overall.to_csv("milk10k_results.csv", index=False)
    per_class_df.to_csv("milk10k_per_class_f1_at_05.csv", index=False)
    cm_df.to_csv("milk10k_confusion_matrix_argmax.csv", index=True)
    Path("milk10k_classification_report_argmax.txt").write_text(report_text, encoding="utf-8")

    plot_confusion(cm, class_names, FIG_DIR / "milk10k_v2__confusion_matrix_argmax")
    plot_per_class_f1(per_class_df, class_names, FIG_DIR / "milk10k_v2__per_class_f1_at_05")

    topc = top_confusions(cm, class_names, k=15)
    topc.to_csv("milk10k_top_confusions.csv", index=False)

    print("MILK10k v2 val metrics:")
    print(overall.to_string(index=False))
    avg_pos = float((probs_sig >= 0.5).sum(axis=1).mean())
    pct_any = float(((probs_sig >= 0.5).sum(axis=1) >= 1).mean())
    pct_multi = float(((probs_sig >= 0.5).sum(axis=1) > 1).mean())
    print(f"Avg predicted positives/sample @0.5: {avg_pos:.3f} | pct>=1: {pct_any:.3f} | pct>1: {pct_multi:.3f}")
    print("\nTop confusions:")
    print(topc.to_string(index=False))


if __name__ == "__main__":
    main()
