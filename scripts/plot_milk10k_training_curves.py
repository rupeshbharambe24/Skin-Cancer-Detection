import json
import re
from pathlib import Path
from typing import Any, Dict, List

import matplotlib.pyplot as plt
import pandas as pd


FIG_DIR = Path("figures")
FIG_DIR.mkdir(exist_ok=True)


def _savefig(base: Path):
    plt.tight_layout()
    plt.savefig(base.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.savefig(base.with_suffix(".pdf"), bbox_inches="tight")
    plt.close()


LINE_RE = re.compile(
    r"^\[(Stage1|Stage2)\]\[(\d+)\/(\d+)\]\s+train=\{'train_loss':\s*([0-9.]+)\}\s+val=\{'macro_f1@0.5':\s*([0-9.]+)\}"
)


def extract_stage_logs(nb_path: Path) -> pd.DataFrame:
    nb: Dict[str, Any] = json.loads(nb_path.read_text(encoding="utf-8"))
    rows: List[Dict[str, object]] = []
    for cell in nb.get("cells", []):
        if cell.get("cell_type") != "code":
            continue
        for out in cell.get("outputs", []):
            if out.get("output_type") not in {"stream", "execute_result", "display_data"}:
                continue
            text = out.get("text", None)
            if text is None:
                continue
            if isinstance(text, list):
                lines = [l.rstrip("\n") for l in text]
            else:
                lines = str(text).splitlines()
            for line in lines:
                m = LINE_RE.match(line.strip())
                if not m:
                    continue
                stage, ep, total, tr_loss, va_f1 = m.groups()
                rows.append(
                    {
                        "stage": stage,
                        "epoch_in_stage": int(ep),
                        "epochs_in_stage": int(total),
                        "train_loss": float(tr_loss),
                        "val_macro_f1@0.5": float(va_f1),
                    }
                )
    if not rows:
        return pd.DataFrame(columns=["stage", "epoch_in_stage", "epochs_in_stage", "train_loss", "val_macro_f1@0.5"])

    df = pd.DataFrame(rows)
    # Create a global epoch index for plotting.
    df["epoch_global"] = 0
    g = 0
    epoch_global = []
    for _, r in df.iterrows():
        epoch_global.append(g)
        g += 1
    df["epoch_global"] = epoch_global
    return df


def plot_curves(df: pd.DataFrame, out_base: Path):
    if df.empty:
        return
    plt.figure(figsize=(8, 4))
    ax1 = plt.gca()
    ax2 = ax1.twinx()

    ax1.plot(df["epoch_global"], df["train_loss"], color="#1f77b4", label="Train loss")
    ax2.plot(df["epoch_global"], df["val_macro_f1@0.5"], color="#d62728", label="Val macro-F1@0.5")

    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Train loss")
    ax2.set_ylabel("Val macro-F1@0.5")
    ax1.set_title("MILK10k Training Curves (Notebook Logs)")

    # Combined legend
    lines, labels = [], []
    for ax in (ax1, ax2):
        l, lab = ax.get_legend_handles_labels()
        lines += l
        labels += lab
    ax1.legend(lines, labels, loc="upper right")
    _savefig(out_base)


def main():
    plt.rcParams.update({"font.size": 10})
    nb_path = Path("MILK10k_ViT_LatentFusion_v2_from_metadata.ipynb")
    df = extract_stage_logs(nb_path)
    df.to_csv("milk10k_training_curves_from_notebook_logs.csv", index=False)
    plot_curves(df, FIG_DIR / "milk10k_v2__training_curves")
    print(f"Extracted {len(df)} epoch log rows")


if __name__ == "__main__":
    main()

