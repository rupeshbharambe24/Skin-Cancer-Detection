import base64
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple


def extract_images(nb_path: Path, out_dir: Path) -> List[Path]:
    nb: Dict[str, Any] = json.loads(nb_path.read_text(encoding="utf-8"))
    out_dir.mkdir(parents=True, exist_ok=True)
    written: List[Path] = []

    for cell_idx, cell in enumerate(nb.get("cells", [])):
        if cell.get("cell_type") != "code":
            continue
        for out_idx, out in enumerate(cell.get("outputs", [])):
            data = out.get("data", {})
            if "image/png" in data:
                b64 = data["image/png"]
                if isinstance(b64, list):
                    b64 = "".join(b64)
                raw = base64.b64decode(b64)
                p = out_dir / f"{nb_path.stem}__cell{cell_idx:03d}__out{out_idx:02d}.png"
                p.write_bytes(raw)
                written.append(p)
            if "application/pdf" in data:
                b64 = data["application/pdf"]
                if isinstance(b64, list):
                    b64 = "".join(b64)
                raw = base64.b64decode(b64)
                p = out_dir / f"{nb_path.stem}__cell{cell_idx:03d}__out{out_idx:02d}.pdf"
                p.write_bytes(raw)
                written.append(p)
    return written


def main():
    out_dir = Path("figures") / "_extracted_from_notebooks"
    out_dir.mkdir(parents=True, exist_ok=True)
    nbs = [
        Path("Skin_Lesions_train.ipynb"),
        Path("Skin_Lesions_train_v2.ipynb"),
        Path("Skin_Lesions_train_v3.ipynb"),
        Path("MILK10k_ViT_LatentFusion_v2_from_metadata.ipynb"),
    ]
    for nb in nbs:
        if not nb.exists():
            continue
        imgs = extract_images(nb, out_dir)
        print(f"{nb}: extracted {len(imgs)} images to {out_dir}")


if __name__ == "__main__":
    main()

