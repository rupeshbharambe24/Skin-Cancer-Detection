from __future__ import annotations

import json
import re
from datetime import date
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.shared import Inches, Pt


ROOT = Path(".")
FIG_DIR = ROOT / "figures"
EXTRACTED_DIR = FIG_DIR / "_extracted_from_notebooks"


def read_text(p: Path) -> str:
    return p.read_text(encoding="utf-8") if p.exists() else ""


def load_stream_text(nb_path: Path) -> str:
    nb: Dict[str, Any] = json.loads(nb_path.read_text(encoding="utf-8"))
    lines: List[str] = []
    for c in nb.get("cells", []):
        if c.get("cell_type") != "code":
            continue
        for o in c.get("outputs", []):
            if o.get("output_type") != "stream":
                continue
            t = o.get("text", "")
            if isinstance(t, list):
                lines.extend([str(x) for x in t])
            else:
                lines.append(str(t))
    return "".join(lines)


def parse_milk10k_notebook_metrics(nb_path: Path) -> Dict[str, float]:
    text = load_stream_text(nb_path)
    keys = [
        "Top-1 Accuracy",
        "Balanced Accuracy",
        "Macro-F1@0.5",
        "Macro-Precision@0.5",
        "Macro-Recall@0.5",
    ]
    out: Dict[str, float] = {}
    for k in keys:
        m = re.search(re.escape(k) + r":\s*([0-9.]+)", text)
        if m:
            out[k] = float(m.group(1))
    # Best Macro-F1@0.5 line (stage logs) can include more digits.
    m = re.search(r"Best Macro-F1@0.5:\s*([0-9.]+)", text)
    if m:
        out["Best Macro-F1@0.5"] = float(m.group(1))
    return out


def parse_milk10k_per_class_f1(nb_path: Path) -> pd.DataFrame:
    text = load_stream_text(nb_path)
    rows = []
    for line in text.splitlines():
        if "F1@0.5" not in line:
            continue
        m = re.match(r"^([A-Z_]{2,7})\s+F1@0.5\s*=\s*([0-9.]+)", line.strip())
        if not m:
            continue
        rows.append({"class": m.group(1), "f1@0.5": float(m.group(2))})
    return pd.DataFrame(rows)


def parse_sklearn_classification_report(report_text: str) -> pd.DataFrame:
    # Parses sklearn classification_report (text) into a table for per-class metrics.
    rows = []
    for line in report_text.splitlines():
        line = line.strip()
        if not line or line.startswith("precision") or line.startswith("accuracy") or line.startswith("macro avg") or line.startswith("weighted avg"):
            continue
        # class precision recall f1 support
        m = re.match(r"^(\S+)\s+([0-9.]+)\s+([0-9.]+)\s+([0-9.]+)\s+(\d+)$", line)
        if m:
            rows.append(
                {
                    "class": m.group(1),
                    "precision": float(m.group(2)),
                    "recall": float(m.group(3)),
                    "f1": float(m.group(4)),
                    "support": int(m.group(5)),
                }
            )
    return pd.DataFrame(rows)


def add_caption(doc: Document, text: str):
    p = doc.add_paragraph(text)
    p.runs[0].italic = True


def add_image(doc: Document, img_path: Path, width_in: float, caption: str):
    if not img_path.exists():
        doc.add_paragraph(f"[TODO] Missing figure: {img_path}")
        return
    doc.add_picture(str(img_path), width=Inches(width_in))
    add_caption(doc, caption)


def add_table(doc: Document, df: pd.DataFrame, title: str, float_fmt: Optional[str] = None):
    doc.add_paragraph(title).runs[0].bold = True
    table = doc.add_table(rows=1, cols=len(df.columns))
    table.style = "Table Grid"
    hdr = table.rows[0].cells
    for j, col in enumerate(df.columns):
        hdr[j].text = str(col)
    for _, row in df.iterrows():
        cells = table.add_row().cells
        for j, col in enumerate(df.columns):
            v = row[col]
            if isinstance(v, float) and float_fmt is not None:
                cells[j].text = format(v, float_fmt)
            else:
                cells[j].text = str(v)


def pick_largest_extracted_image(prefix: str) -> Optional[Path]:
    cand = list(EXTRACTED_DIR.glob(f"{prefix}__*.png"))
    if not cand:
        return None
    cand.sort(key=lambda p: p.stat().st_size, reverse=True)
    return cand[0]


def main():
    out_docx = ROOT / "Results_and_Discussion_Report.docx"

    # Load computed evaluation tables (binary + milk10k recomputed)
    binary_df = pd.read_csv(ROOT / "binary_results.csv") if (ROOT / "binary_results.csv").exists() else pd.DataFrame()
    milk10k_re_df = pd.read_csv(ROOT / "milk10k_results.csv") if (ROOT / "milk10k_results.csv").exists() else pd.DataFrame()

    # MILK10k notebook-reported metrics (saved outputs in notebook file)
    milk_nb_metrics = parse_milk10k_notebook_metrics(ROOT / "MILK10k_ViT_LatentFusion_v2_from_metadata.ipynb")
    milk_nb_perclass_f1 = parse_milk10k_per_class_f1(ROOT / "MILK10k_ViT_LatentFusion_v2_from_metadata.ipynb")

    # Binary dataset counts
    counts_path = FIG_DIR / "binary_dataset_counts.csv"
    counts_df = pd.read_csv(counts_path) if counts_path.exists() else pd.DataFrame()

    # MILK10k report artifacts
    milk_report_txt = load_stream_text(ROOT / "MILK10k_ViT_LatentFusion_v2_from_metadata.ipynb")
    # Extract the argmax classification_report block from the notebook output for per-class table.
    # We'll take the first block that follows "Multiclass classification report (argmax):"
    report_block = ""
    m = re.search(r"Multiclass classification report \(argmax\):\s*\n(.*?)(\n\n|\Z)", milk_report_txt, flags=re.S)
    if m:
        report_block = m.group(1).strip()
    milk_argmax_df = parse_sklearn_classification_report(report_block) if report_block else pd.DataFrame()

    doc = Document()

    # Typography
    style = doc.styles["Normal"]
    style.font.name = "Calibri"
    style.font.size = Pt(11)

    title = doc.add_paragraph("Results & Discussion Report")
    title.runs[0].bold = True
    title.runs[0].font.size = Pt(18)
    title.alignment = WD_ALIGN_PARAGRAPH.CENTER
    doc.add_paragraph(f"Generated on {date.today().isoformat()} (local workspace evaluation + notebook outputs).").alignment = WD_ALIGN_PARAGRAPH.CENTER

    # 1. Experimental Setup
    doc.add_heading("1. Experimental Setup", level=1)

    doc.add_heading("1.1 Hardware & Environment", level=2)
    env_info = read_text(ROOT / "env_info.txt").strip()
    if env_info:
        doc.add_paragraph(env_info)
    else:
        doc.add_paragraph("[TODO] env_info.txt not found.")

    doc.add_heading("1.2 Datasets", level=2)
    # MILK10k
    doc.add_paragraph(
        "MILK10k (ISIC) multimodal dataset: paired clinical close-up and dermoscopic images per lesion with 11 benchmark diagnostic categories. "
        "The benchmark emphasizes class-balanced performance via Macro-F1 (Dice) across categories."
    )
    doc.add_paragraph(
        "This project’s MILK10k notebook performs a lesion-level stratified split (15% validation) using StratifiedShuffleSplit with random_state=42, "
        "ensuring paired views remain together per lesion."
    )
    doc.add_paragraph(
        "Metadata fields used in the notebook include age, sex, and anatomical site; skin tone and MONET concepts are present in MILK10k supplements but were not enabled "
        "in the v2 notebook configuration (extra_meta_file=None), therefore MONET auxiliary targets are unused (monet_dim=0; head_monet=None)."
    )
    # Kaggle binary dataset
    if not counts_df.empty:
        total_train = int(counts_df[counts_df["split"] == "train"]["count"].sum())
        total_test = int(counts_df[counts_df["split"] == "test"]["count"].sum())
        doc.add_paragraph(
            f"Kaggle-style binary dataset (benign vs malignant): directory-structured ImageFolder dataset under `data/Skin Lesions Classification/` "
            f"with {total_train} training images and {total_test} test images. Within-notebook validation is a 20% split of the training folder (seed=42)."
        )
        add_table(doc, counts_df, "Binary Dataset Image Counts (from folder scan):")
    else:
        doc.add_paragraph("[TODO] Could not load binary_dataset_counts.csv.")

    doc.add_heading("1.3 Preprocessing & Augmentation", level=2)
    doc.add_paragraph("Binary notebooks (224x224) train-time augmentation (as implemented):")
    doc.add_paragraph("Skin_Lesions_train: RandomResizedCrop, RandomHorizontalFlip, RandomRotation(20), Normalize(mean=0.5,std=0.5).")
    doc.add_paragraph("Skin_Lesions_train_v2: RandomResizedCrop, RandomHorizontalFlip, RandomRotation(20), ColorJitter, ImageNet normalization.")
    doc.add_paragraph("Skin_Lesions_train_v3: RandomResizedCrop, RandomHorizontalFlip, RandomRotation(10), light ColorJitter, ImageNet normalization.")
    doc.add_paragraph(
        "Important evaluation note: In Skin_Lesions_train and Skin_Lesions_train_v2, validation is created via `random_split` on a dataset already bound to "
        "the training transform, which causes random augmentation to be applied on validation. For this report, test-set evaluation uses deterministic resize+normalize "
        "transforms to avoid validation/test augmentation."
    )
    doc.add_paragraph("MILK10k v2 transforms: train uses resize + flip + mild color jitter/blur; validation uses deterministic resize+normalize only.")

    doc.add_heading("1.4 Models Trained (as implemented)", level=2)
    doc.add_paragraph(
        "MILK10k model (v2 notebook): Dual-view DINOv2 ViT-S/14 backbone (torch.hub), LoRA adapters applied to attention qkv/proj Linear layers with modality-specific branches "
        "(clinical vs dermoscopic). Tokens from both views are fused with a Latent Cross-Attention fuser (8 learnable latents, 3 fusion layers, 6 heads). "
        "Metadata is embedded as additional tokens (age/sex/site/tone embeddings; MONET optional). Two heads are trained jointly: (i) a sigmoid head for threshold-based macro-F1 "
        "and (ii) a softmax head for standard argmax multiclass accuracy; an exclusivity penalty encourages near-single-label sigmoid outputs."
    )
    doc.add_paragraph(
        "Binary models: (i) a custom CNN baseline (4 conv blocks + MLP) trained from scratch (Skin_Lesions_train), "
        "(ii) ResNet18 transfer learning (Skin_Lesions_train_v2) with a frozen backbone and a dropout+linear classifier head, "
        "(iii) ResNet18 transfer learning with stratified split and tuned augmentation (Skin_Lesions_train_v3)."
    )

    doc.add_heading("1.5 Training Protocol", level=2)
    doc.add_paragraph("Binary (common): batch_size=32; CrossEntropyLoss; early stopping with patience (v1=10, v2=8, v3=10).")
    doc.add_paragraph("Binary v1 optimizer: Adam(lr=1e-3), epochs=100.")
    doc.add_paragraph("Binary v2/v3 optimizer: AdamW with ReduceLROnPlateau scheduler; epochs v2=50, v3=60.")
    doc.add_paragraph(
        "MILK10k (v2 notebook): two-stage training with gradient accumulation (micro_batch=4, grad_accum=8), AdamW with parameter groups (heads/LoRA/unfreeze), "
        "cosine LR schedule with warmup, AMP enabled when CUDA is available (cfg.use_amp=True). Stage1 epochs=12; Stage2 epochs=30."
    )

    # 2. Evaluation Metrics
    doc.add_heading("2. Evaluation Metrics", level=1)
    doc.add_paragraph(
        "MILK10k (11 classes): Top-1 Accuracy and Balanced Accuracy are computed from argmax predictions. "
        "Macro-Precision/Recall/F1@0.5 are computed by thresholding predicted probabilities at 0.5 per class and averaging per-class scores (multi-label style), "
        "consistent with the notebook’s metric implementation. The MILK10k benchmark’s primary reported metric is Macro-F1 (Dice) across diagnostic categories."
    )
    doc.add_paragraph(
        "Binary (2 classes): report Test Accuracy, Precision, Recall, F1 (malignant as positive class), ROC-AUC, and PR-AUC computed on the held-out test folder."
    )

    # 3. Results: Binary
    doc.add_heading("3. Results: Binary Classification (Kaggle)", level=1)

    doc.add_heading("3.1 Summary Table", level=2)
    if not binary_df.empty and "status" in binary_df.columns:
        ok = binary_df[binary_df["status"] == "ok"].copy()
        cols = [
            "model_name",
            "test_accuracy",
            "test_precision",
            "test_recall",
            "test_f1",
            "test_roc_auc",
            "test_pr_auc",
        ]
        ok = ok[cols]
        add_table(doc, ok, "Binary Test Metrics (computed from saved weights on the test split):", float_fmt=".4f")
    else:
        doc.add_paragraph("[TODO] binary_results.csv missing or incomplete.")

    doc.add_heading("3.2 Plots", level=2)
    # Training curves from notebook-extracted plots
    v1_curve = pick_largest_extracted_image("Skin_Lesions_train")
    v2_curve = pick_largest_extracted_image("Skin_Lesions_train_v2")
    v3_curve = pick_largest_extracted_image("Skin_Lesions_train_v3")
    if v1_curve:
        add_image(doc, v1_curve, 6.5, "Figure 1. Training curves (Accuracy/Loss) from Skin_Lesions_train.ipynb (notebook output).")
    if v2_curve:
        add_image(doc, v2_curve, 6.5, "Figure 2. Training curves (Accuracy/Loss) from Skin_Lesions_train_v2.ipynb (notebook output).")
    if v3_curve:
        add_image(doc, v3_curve, 6.5, "Figure 3. Training curves (Accuracy/Loss) from Skin_Lesions_train_v3.ipynb (notebook output).")

    # Confusion/ROC/PR for each binary model
    for key, name in [
        ("binary_custom_cnn_v1", "Custom CNN (v1)"),
        ("binary_resnet18_v2", "ResNet18 (v2)"),
        ("binary_resnet18_v3", "ResNet18 (v3)"),
    ]:
        add_image(
            doc,
            FIG_DIR / f"{key}__confusion_matrix.png",
            5.5,
            f"Figure. Confusion matrix on the test split ({name}).",
        )
        add_image(doc, FIG_DIR / f"{key}__roc.png", 5.5, f"Figure. ROC curve on the test split ({name}).")
        add_image(doc, FIG_DIR / f"{key}__pr.png", 5.5, f"Figure. Precision–Recall curve on the test split ({name}).")

    # Failure cases (optional)
    add_image(doc, FIG_DIR / "binary_resnet18_v2__failure_cases.png", 6.5, "Figure. Representative failure cases (ResNet18 v2, test).")

    doc.add_heading("3.3 Discussion (binary)", level=2)
    doc.add_paragraph(
        "Across the binary task, transfer learning with ResNet18 improves separability relative to the shallow custom CNN by leveraging pretrained low-level texture and color features. "
        "The PR-AUC is especially informative under class imbalance; improvements here indicate better malignant retrieval at clinically relevant operating points. "
        "The v1/v2 notebooks’ validation curves should be interpreted cautiously due to validation-time random augmentation in those implementations; test metrics reported above "
        "use deterministic preprocessing and are therefore the recommended comparison basis."
    )

    # 4. Results: MILK10k
    doc.add_heading("4. Results: Multiclass (MILK10k, 11 classes)", level=1)

    doc.add_heading("4.1 Summary Metrics (must print exact values)", level=2)
    # Notebook-saved metrics
    if milk_nb_metrics:
        nb_row = {
            "source": "Notebook saved outputs (MILK10k v2)",
            "Top-1 Accuracy": milk_nb_metrics.get("Top-1 Accuracy", None),
            "Balanced Accuracy": milk_nb_metrics.get("Balanced Accuracy", None),
            "Macro-Precision@0.5": milk_nb_metrics.get("Macro-Precision@0.5", None),
            "Macro-Recall@0.5": milk_nb_metrics.get("Macro-Recall@0.5", None),
            "Macro-F1@0.5": milk_nb_metrics.get("Macro-F1@0.5", None),
        }
        summary_rows = [nb_row]
    else:
        summary_rows = []

    # Re-evaluated metrics (current checkpoint on disk, computed in scripts/eval_milk10k_v2.py)
    if not milk10k_re_df.empty:
        r = milk10k_re_df.iloc[0].to_dict()
        summary_rows.append(
            {
                "source": "Re-evaluated (current best_milk10k_vit_latentfusion.pt)",
                "Top-1 Accuracy": r.get("val_top1_accuracy", None),
                "Balanced Accuracy": r.get("val_balanced_accuracy", None),
                "Macro-Precision@0.5": r.get("val_macro_precision@0.5", None),
                "Macro-Recall@0.5": r.get("val_macro_recall@0.5", None),
                "Macro-F1@0.5": r.get("val_macro_f1@0.5", None),
            }
        )

    if summary_rows:
        add_table(doc, pd.DataFrame(summary_rows), "MILK10k Summary Metrics:", float_fmt=".4f")
    else:
        doc.add_paragraph("[TODO] MILK10k metrics unavailable.")

    doc.add_heading("4.2 Per-class Results", level=2)
    if not milk_argmax_df.empty:
        add_table(doc, milk_argmax_df, "Per-class Precision/Recall/F1/Support (argmax classification_report from notebook output):", float_fmt=".4f")
    else:
        doc.add_paragraph("[TODO] Could not parse per-class argmax classification report from notebook outputs.")

    if not milk_nb_perclass_f1.empty:
        add_table(doc, milk_nb_perclass_f1, "Per-class F1@0.5 (thresholded, from notebook output):", float_fmt=".4f")
    else:
        doc.add_paragraph("[TODO] Could not parse per-class F1@0.5 from notebook output.")

    # Include our regenerated bar chart (from current checkpoint) and/or notebook-extracted plots
    add_image(doc, FIG_DIR / "milk10k_v2__per_class_f1_at_05.png", 6.5, "Figure. Per-class F1@0.5 bar chart (recomputed from current checkpoint).")

    doc.add_heading("4.3 Confusion Matrix & Error Analysis", level=2)
    # Prefer notebook-extracted confusion matrix image for consistency with saved notebook outputs.
    nb_conf = EXTRACTED_DIR / "MILK10k_ViT_LatentFusion_v2_from_metadata__cell046__out01.png"
    add_image(doc, nb_conf, 6.5, "Figure. Confusion matrix (argmax) as rendered in the MILK10k v2 notebook output.")

    # Top confusions from recomputed confusion matrix
    topc_path = ROOT / "milk10k_top_confusions.csv"
    if topc_path.exists():
        topc = pd.read_csv(topc_path)
        add_table(doc, topc.head(10), "Top Confusions (counts from re-evaluated checkpoint confusion matrix):")
    else:
        doc.add_paragraph("[TODO] milk10k_top_confusions.csv missing.")

    doc.add_heading("4.4 Curves", level=2)
    add_image(doc, FIG_DIR / "milk10k_v2__training_curves.png", 6.5, "Figure. MILK10k training curves extracted from notebook stage logs (train loss vs val macro-F1@0.5).")

    doc.add_heading("4.5 Discussion (MILK10k)", level=2)
    doc.add_paragraph(
        "The MILK10k task is strongly long-tailed; consequently, Top-1 Accuracy can remain relatively high while class-balanced metrics degrade, "
        "particularly on rare categories (e.g., BEN_OTH and MAL_OTH). In the notebook-reported run, these rare classes exhibit F1@0.5=0, indicating "
        "insufficient effective positive evidence or overly conservative thresholding. The confusion matrix highlights clinically plausible confusions such as MEL↔NV and BCC↔SCCKA/BKL."
    )
    doc.add_paragraph(
        "Re-evaluation of the checkpoint currently present in the workspace produced markedly different threshold-based macro metrics due to many classes exceeding the 0.5 threshold per sample "
        "(multi-positive predictions), suggesting probability miscalibration or a mismatch between the checkpoint file and the notebook-saved run. "
        "For benchmarking, checkpoint versioning (e.g., unique filenames per run + hash logging) is recommended to ensure consistency between reported numbers and persisted weights."
    )
    doc.add_paragraph(
        "Although MILK10k provides MONET concept probabilities and related metadata in supplements, the v2 notebook configuration did not enable loading these features. "
        "In future runs, explicitly incorporating MONET/ITA features as auxiliary inputs or targets may help normalize for skin tone and improve minority-class performance."
    )

    # 5. Comparison to Reference Papers (optional)
    doc.add_heading("5. Comparison to Reference Papers (optional, only if you can cite accurately)", level=1)
    doc.add_paragraph(
        "Not included in this report due to limited metric/task comparability across available PDFs. "
        "If a specific reference and evaluation protocol are selected, a controlled comparison table can be added."
    )

    # 6. Reproducibility Checklist
    doc.add_heading("6. Reproducibility Checklist (must include)", level=1)
    doc.add_paragraph("Seeds: binary notebooks set torch/numpy seed=42; MILK10k split uses random_state=42 (lesion-level StratifiedShuffleSplit).")
    doc.add_paragraph("Splits: MILK10k split is lesion-level; binary split is per-image within the training folder (20% validation).")
    doc.add_paragraph(
        "Data leakage checks: paired MILK10k views are kept within-lesion; binary validation augmentation issue identified for v1/v2 (random aug applied on validation)."
    )
    doc.add_paragraph("Code versioning: results depend on the exact persisted weight files (`models/*.pth`, `best_milk10k_vit_latentfusion.pt`).")
    doc.add_paragraph(
        "Hardware: this report was generated in a CPU-only PyTorch environment; training was not re-run, only inference/evaluation and notebook-output parsing."
    )
    doc.add_paragraph(
        "Medical imaging limitations: rare-class supports are small and metrics can be unstable; threshold-based scoring is sensitive to calibration; reported results should be "
        "validated on external cohorts before clinical use."
    )

    # References (required citations)
    doc.add_paragraph("References").runs[0].bold = True
    refs = [
        "[1] ISIC Challenge: MILK10k. Primary metric described as Macro-F1 (Dice). https://www.isic-archive.com/#!/challenge/milk10k",
        "[2] ISIC Archive: MILK10k dataset entry (includes metadata + supplements; MONET concepts referenced). https://www.isic-archive.com/#!/topWithHeader/onlyHeaderTop/gallery?filter=%5B%22dataset%22%2C%22MILK10k%22%5D",
        "[3] CLAIM guideline (Checklist for Artificial Intelligence in Medical Imaging) for reproducibility reporting. https://pubs.rsna.org/doi/10.1148/ryai.2020200029",
    ]
    for r in refs:
        doc.add_paragraph(r)
    doc.add_paragraph(f"Accessed: {date.today().isoformat()}.")

    doc.save(out_docx)
    print(f"Wrote: {out_docx}")


if __name__ == "__main__":
    main()

