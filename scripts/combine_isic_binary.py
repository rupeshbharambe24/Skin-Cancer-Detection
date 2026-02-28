import csv
import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split


DATASET_A_ROOT = Path(r"D:\Work-Projects\Skin Cancer Detection\data\Skin Lesions Classification")
DATASET_B_ROOT = Path(r"D:\Work-Projects\Skin Cancer Detection\data\ISIC_2019-20 Dataset")
OUT_ROOT = Path(r"D:\Work-Projects\Skin Cancer Detection\combined_isic_binary")

SEED = 42
SPLIT = (0.70, 0.15, 0.15)
PHASH_HAMMING_THRESHOLD = 5


@dataclass
class ImageRec:
    src_path: Path
    rel_path: str
    label: int
    label_name: str
    source: str
    orig_id: str
    patient_id: str
    sha256: str
    phash: str
    width: int
    height: int
    file_size: int


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def dhash(p: Path, size: int = 8) -> str:
    # 64-bit dHash (hex string length 16)
    img = Image.open(p).convert("L").resize((size + 1, size), Image.Resampling.LANCZOS)
    pixels = np.asarray(img)
    diff = pixels[:, 1:] > pixels[:, :-1]
    bits = diff.flatten().astype(np.uint8)
    val = 0
    for b in bits:
        val = (val << 1) | int(b)
    return f"{val:016x}"


def hamming_hex(a: str, b: str) -> int:
    return bin(int(a, 16) ^ int(b, 16)).count("1")


def detect_label_from_path(p: Path) -> Optional[Tuple[int, str]]:
    parts = [s.lower() for s in p.parts]
    if "benign" in parts:
        return 0, "benign"
    if "malignant" in parts:
        return 1, "malignant"
    return None


def collect_images(root: Path, source: str) -> List[ImageRec]:
    out: List[ImageRec] = []
    for p in root.rglob("*"):
        if not p.is_file():
            continue
        if p.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
            continue
        label = detect_label_from_path(p)
        if label is None:
            continue
        lab, lab_name = label
        rel = p.name
        orig_id = p.stem
        patient_id = ""  # no metadata in folders
        try:
            with Image.open(p) as im:
                w, h = im.size
        except Exception:
            w, h = 0, 0
        rec = ImageRec(
            src_path=p,
            rel_path=rel,
            label=lab,
            label_name=lab_name,
            source=source,
            orig_id=orig_id,
            patient_id=patient_id,
            sha256=sha256_file(p),
            phash=dhash(p),
            width=w,
            height=h,
            file_size=p.stat().st_size,
        )
        out.append(rec)
    return out


def bucket_keys(phash_hex: str) -> List[str]:
    # 64-bit hash -> 4 chunks of 16 bits each
    return [
        phash_hex[0:4],
        phash_hex[4:8],
        phash_hex[8:12],
        phash_hex[12:16],
    ]


def build_near_duplicate_groups(records: List[ImageRec]) -> Tuple[List[List[int]], Dict[int, int]]:
    # return groups of indices, and index->group id
    buckets: Dict[str, List[int]] = {}
    for i, r in enumerate(records):
        for k in bucket_keys(r.phash):
            buckets.setdefault(k, []).append(i)

    visited = set()
    groups: List[List[int]] = []
    idx_to_group: Dict[int, int] = {}

    for i, r in enumerate(records):
        if i in visited:
            continue
        # candidate set from buckets
        cand = set()
        for k in bucket_keys(r.phash):
            cand.update(buckets.get(k, []))
        cand.discard(i)

        group = [i]
        visited.add(i)
        for j in cand:
            if j in visited:
                continue
            d = hamming_hex(r.phash, records[j].phash)
            if d <= PHASH_HAMMING_THRESHOLD:
                group.append(j)
                visited.add(j)
        if len(group) > 1:
            gid = len(groups)
            for idx in group:
                idx_to_group[idx] = gid
            groups.append(group)
    return groups, idx_to_group


def choose_keep(records: List[ImageRec], group: List[int]) -> int:
    # prefer higher resolution, then larger file size, then patient_id present (none here)
    def score(idx: int) -> Tuple[int, int, int]:
        r = records[idx]
        return (r.width * r.height, r.file_size, 1 if r.patient_id else 0)

    best = max(group, key=score)
    return best


def ensure_out_dirs():
    (OUT_ROOT / "images").mkdir(parents=True, exist_ok=True)


def copy_or_link(src: Path, dst: Path) -> str:
    try:
        os.symlink(src, dst)
        return "symlink"
    except Exception:
        dst.write_bytes(src.read_bytes())
        return "copy"


def main():
    np.random.seed(SEED)
    ensure_out_dirs()

    recs = []
    recs += collect_images(DATASET_A_ROOT, "fanconic")
    recs += collect_images(DATASET_B_ROOT, "sallyibrahim")

    # exact duplicates by sha256
    by_sha: Dict[str, List[int]] = {}
    for i, r in enumerate(recs):
        by_sha.setdefault(r.sha256, []).append(i)

    drop = set()
    dedup_rows = []
    for sha, idxs in by_sha.items():
        if len(idxs) <= 1:
            continue
        keep = choose_keep(recs, idxs)
        for i in idxs:
            if i == keep:
                continue
            drop.add(i)
            dedup_rows.append(
                {
                    "dup_type": "sha256",
                    "keep": str(recs[keep].src_path),
                    "drop": str(recs[i].src_path),
                    "reason": "exact_duplicate",
                }
            )

    # near duplicates by phash (on remaining indices)
    remaining = [i for i in range(len(recs)) if i not in drop]
    rem_records = [recs[i] for i in remaining]
    groups, idx_to_group = build_near_duplicate_groups(rem_records)

    groups_out = []
    for g in groups:
        # map back to original indices
        orig_group = [remaining[i] for i in g]
        keep = choose_keep(recs, orig_group)
        group_paths = [str(recs[i].src_path) for i in orig_group]
        groups_out.append({"keep": str(recs[keep].src_path), "members": group_paths})
        for i in orig_group:
            if i == keep or i in drop:
                continue
            drop.add(i)
            dedup_rows.append(
                {
                    "dup_type": "phash",
                    "keep": str(recs[keep].src_path),
                    "drop": str(recs[i].src_path),
                    "reason": f"phash<= {PHASH_HAMMING_THRESHOLD}",
                }
            )

    # build final list
    final = [r for i, r in enumerate(recs) if i not in drop]

    # copy/symlink images
    manifest_rows = []
    for r in final:
        out_name = f"{r.source}__{r.src_path.name}"
        out_path = OUT_ROOT / "images" / out_name
        if not out_path.exists():
            mode = copy_or_link(r.src_path, out_path)
        else:
            mode = "exists"
        manifest_rows.append(
            {
                "filepath": str(Path("images") / out_name).replace("\\", "/"),
                "label": r.label,
                "label_name": r.label_name,
                "source": r.source,
                "orig_id": r.orig_id,
                "patient_id": r.patient_id,
                "sha256": r.sha256,
                "phash": r.phash,
                "width": r.width,
                "height": r.height,
                "file_size": r.file_size,
                "link_mode": mode,
            }
        )

    # write manifest
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    manifest_path = OUT_ROOT / "manifest.csv"
    with manifest_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(manifest_rows[0].keys()))
        w.writeheader()
        w.writerows(manifest_rows)

    # write dedup reports
    dedup_path = OUT_ROOT / "dedup_report.csv"
    if dedup_rows:
        with dedup_path.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(dedup_rows[0].keys()))
            w.writeheader()
            w.writerows(dedup_rows)
    else:
        dedup_path.write_text("dup_type,keep,drop,reason\n", encoding="utf-8")

    (OUT_ROOT / "dedup_groups.json").write_text(json.dumps(groups_out, indent=2), encoding="utf-8")

    # split
    labels = np.array([r["label"] for r in manifest_rows])
    idx = np.arange(len(manifest_rows))
    train_idx, temp_idx = train_test_split(
        idx, test_size=1 - SPLIT[0], stratify=labels, random_state=SEED
    )
    temp_labels = labels[temp_idx]
    val_size = SPLIT[1] / (SPLIT[1] + SPLIT[2])
    val_idx, test_idx = train_test_split(
        temp_idx, test_size=1 - val_size, stratify=temp_labels, random_state=SEED
    )

    def write_split(name: str, indices: np.ndarray):
        p = OUT_ROOT / f"{name}.csv"
        with p.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(manifest_rows[0].keys()))
            w.writeheader()
            w.writerows([manifest_rows[i] for i in indices])

    write_split("train", train_idx)
    write_split("val", val_idx)
    write_split("test", test_idx)

    # summary
    def count(idx_list):
        labs = labels[idx_list]
        return int((labs == 0).sum()), int((labs == 1).sum())

    b0, m0 = count(train_idx)
    b1, m1 = count(val_idx)
    b2, m2 = count(test_idx)

    print("Total images:", len(manifest_rows))
    print("Train:", len(train_idx), "benign", b0, "malignant", m0)
    print("Val:", len(val_idx), "benign", b1, "malignant", m1)
    print("Test:", len(test_idx), "benign", b2, "malignant", m2)
    print("Manifest:", manifest_path)
    print("Dedup report:", dedup_path)


if __name__ == "__main__":
    main()

