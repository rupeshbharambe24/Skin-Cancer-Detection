import json
from pathlib import Path


SRC = Path("Skin_Lesions_train_custom.ipynb")
DST = Path("Combined_Binary_CustomCNN_Train.ipynb")


NEW_DATASET_CELL = """\
# Combined manifest-based dataset (no random aug on val/test)
from pathlib import Path

OUT_ROOT = Path("combined_isic_binary")
train_csv = OUT_ROOT / "train.csv"
val_csv = OUT_ROOT / "val.csv"
test_csv = OUT_ROOT / "test.csv"

train_df = pd.read_csv(train_csv)
val_df = pd.read_csv(val_csv)
test_df = pd.read_csv(test_csv)

class_names = ["benign", "malignant"]
class_to_idx = {c: i for i, c in enumerate(class_names)}

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

train_data = ManifestDataset(train_df, OUT_ROOT, transform=train_transforms)
val_data = ManifestDataset(val_df, OUT_ROOT, transform=val_test_transforms)
test_data = ManifestDataset(test_df, OUT_ROOT, transform=val_test_transforms)

train_targets = train_df["label"].values
print("Train/Val/Test:", len(train_data), len(val_data), len(test_data))
print("Train counts:", train_df["label_name"].value_counts())
"""


AMP_CELL = """\
# AMP + deterministic seeds
use_amp = torch.cuda.is_available()
scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

seed_everything(42)
"""


EVAL_CELL = """\
# Evaluation + plots + save outputs
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix,
    roc_curve, precision_recall_curve
)
import json

OUT_FIG = OUT_ROOT / "figures"
OUT_FIG.mkdir(parents=True, exist_ok=True)

model.eval()
all_probs = []
all_labels = []

with torch.no_grad():
    for imgs, labels in test_loader:
        imgs = imgs.to(device)
        labels = labels.to(device)
        with torch.cuda.amp.autocast(enabled=use_amp):
            logits = model(imgs)
        probs = torch.softmax(logits, dim=1)[:, 1]  # malignant prob
        all_probs.append(probs.cpu().numpy())
        all_labels.append(labels.cpu().numpy())

probs = np.concatenate(all_probs)
y_true = np.concatenate(all_labels)
y_pred = (probs >= 0.5).astype(int)

acc = accuracy_score(y_true, y_pred)
prec = precision_score(y_true, y_pred, zero_division=0)
rec = recall_score(y_true, y_pred, zero_division=0)
f1 = f1_score(y_true, y_pred, zero_division=0)
roc_auc = roc_auc_score(y_true, probs)
pr_auc = average_precision_score(y_true, probs)

print(f"Accuracy: {acc:.4f}")
print(f"Precision: {prec:.4f}")
print(f"Recall: {rec:.4f}")
print(f"F1: {f1:.4f}")
print(f"ROC-AUC: {roc_auc:.4f}")
print(f"PR-AUC: {pr_auc:.4f}")

# Confusion matrix
cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(4,4))
plt.imshow(cm, cmap="Blues")
plt.title("Confusion Matrix (Test)")
plt.xlabel("Predicted")
plt.ylabel("True")
for i in range(2):
    for j in range(2):
        plt.text(j, i, str(cm[i, j]), ha="center", va="center")
plt.tight_layout()
plt.savefig(OUT_FIG / "confusion_matrix.png", dpi=300)
plt.close()

# ROC
fpr, tpr, _ = roc_curve(y_true, probs)
plt.figure(figsize=(4,4))
plt.plot(fpr, tpr, label=f"AUC={roc_auc:.4f}")
plt.plot([0,1],[0,1],'--',color='gray')
plt.xlabel("FPR"); plt.ylabel("TPR"); plt.title("ROC Curve")
plt.legend()
plt.tight_layout()
plt.savefig(OUT_FIG / "roc_curve.png", dpi=300)
plt.close()

# PR
pr, rc, _ = precision_recall_curve(y_true, probs)
plt.figure(figsize=(4,4))
plt.plot(rc, pr, label=f"AP={pr_auc:.4f}")
plt.xlabel("Recall"); plt.ylabel("Precision"); plt.title("PR Curve")
plt.legend()
plt.tight_layout()
plt.savefig(OUT_FIG / "pr_curve.png", dpi=300)
plt.close()

# Save outputs
(OUT_ROOT / "preds_test.csv").write_text(
    "prob_malignant,label\\n" + "\\n".join([f\"{p:.6f},{y}\" for p,y in zip(probs, y_true)]),
    encoding="utf-8"
)

results = {
    "accuracy": float(acc),
    "precision": float(prec),
    "recall": float(rec),
    "f1": float(f1),
    "roc_auc": float(roc_auc),
    "pr_auc": float(pr_auc),
}
(OUT_ROOT / "results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
"""


def main():
    nb = json.loads(SRC.read_text(encoding="utf-8"))
    cells = nb["cells"]

    # Insert AMP cell after first import block
    insert_idx = 0
    for i, c in enumerate(cells):
        if c.get("cell_type") != "code":
            continue
        src = "".join(c.get("source", []))
        if "import" in src and "torch" in src:
            insert_idx = i + 1
            break
    cells.insert(insert_idx, {"cell_type": "code", "metadata": {}, "source": AMP_CELL.splitlines(keepends=True), "outputs": [], "execution_count": None})

    # Replace dataset cell
    replaced = False
    for i, c in enumerate(cells):
        if c.get("cell_type") != "code":
            continue
        src = "".join(c.get("source", []))
        if "full_train_dataset = datasets.ImageFolder" in src and "train_idx" in src:
            cells[i]["source"] = NEW_DATASET_CELL.splitlines(keepends=True)
            cells[i]["outputs"] = []
            cells[i]["execution_count"] = None
            replaced = True
            break

    if not replaced:
        raise SystemExit("Could not find dataset cell to replace.")

    # Append eval cell at end
    cells.append({"cell_type": "code", "metadata": {}, "source": EVAL_CELL.splitlines(keepends=True), "outputs": [], "execution_count": None})

    DST.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
    print("Wrote:", DST)


if __name__ == "__main__":
    main()

