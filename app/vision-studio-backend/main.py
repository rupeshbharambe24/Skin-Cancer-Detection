from __future__ import annotations

import io
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
from torchvision import transforms

from model_registry import ModelSpec, count_parameters, get_model_specs

app = FastAPI(title="Vision Studio Backend", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL_SPECS: Dict[str, ModelSpec] = {spec.key: spec for spec in get_model_specs()}
MODEL_CACHE: Dict[Tuple[str, str], torch.nn.Module] = {}


def build_transform(spec: ModelSpec) -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize((spec.input_size, spec.input_size)),
            transforms.ToTensor(),
            transforms.Normalize(spec.norm_mean, spec.norm_std),
        ]
    )


def resolve_device(requested: str) -> torch.device:
    if requested == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def load_model(spec: ModelSpec, device: torch.device) -> torch.nn.Module:
    cache_key = (spec.key, device.type)
    if cache_key in MODEL_CACHE:
        return MODEL_CACHE[cache_key]
    if not spec.weights_path.exists():
        raise FileNotFoundError(str(spec.weights_path))
    model = spec.builder()
    state = torch.load(spec.weights_path, map_location="cpu")
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    MODEL_CACHE[cache_key] = model
    return model


def clear_cache(model_id: Optional[str] = None) -> None:
    if model_id is None:
        MODEL_CACHE.clear()
        return
    keys = [key for key in MODEL_CACHE if key[0] == model_id]
    for key in keys:
        MODEL_CACHE.pop(key, None)


def run_inference(
    model_id: str,
    image: Image.Image,
    device_obj: torch.device,
) -> Dict[str, object]:
    spec = MODEL_SPECS[model_id]
    if not spec.weights_path.exists():
        raise HTTPException(status_code=400, detail=f"Weights missing for {model_id}")
    transform = build_transform(spec)
    tensor = transform(image).unsqueeze(0).to(device_obj)
    model = load_model(spec, device_obj)
    with torch.no_grad():
        logits = model(tensor)
        probs = torch.softmax(logits, dim=1).cpu().numpy().flatten()
    class_names = spec.class_names
    top_idx = int(np.argmax(probs))
    return {
        "model_id": model_id,
        "label": class_names[top_idx],
        "confidence": float(probs[top_idx]),
        "probs": {name: float(prob) for name, prob in zip(class_names, probs)},
        "device": device_obj.type,
    }


@app.get("/api/health")
def health() -> Dict[str, str]:
    return {"status": "ok"}


@app.get("/")
def root() -> Dict[str, str]:
    return {
        "status": "backend_running",
        "frontend": "http://127.0.0.1:5173",
        "docs": "http://127.0.0.1:8000/docs",
    }


@app.get("/api/models")
def list_models() -> Dict[str, list]:
    items = []
    for spec in MODEL_SPECS.values():
        model = spec.builder()
        items.append(
            {
                "id": spec.key,
                "name": spec.name,
                "params": count_parameters(model),
                "input_size": spec.input_size,
                "class_names": spec.class_names,
                "weights_path": str(spec.weights_path),
                "weights_found": spec.weights_path.exists(),
                "test_accuracy": spec.test_accuracy,
            }
        )
    return {"models": items}


@app.post("/api/reload")
def reload_models(model_id: Optional[str] = Form(default=None)) -> Dict[str, str]:
    if model_id is not None and model_id not in MODEL_SPECS:
        raise HTTPException(status_code=404, detail="Unknown model_id")
    clear_cache(model_id=model_id)
    return {"status": "reloaded"}


@app.post("/api/predict")
async def predict(
    model_id: str = Form(...),
    device: str = Form("cpu"),
    file: UploadFile = File(...),
) -> Dict[str, object]:
    spec = MODEL_SPECS.get(model_id)
    if spec is None:
        raise HTTPException(status_code=404, detail="Unknown model_id")
    if not spec.weights_path.exists():
        raise HTTPException(status_code=400, detail="Weights file not found")

    device_obj = resolve_device(device)

    raw = await file.read()
    try:
        image = Image.open(io.BytesIO(raw)).convert("RGB")
    except Exception as exc:
        raise HTTPException(status_code=400, detail="Invalid image file") from exc

    return run_inference(model_id, image, device_obj)


@app.post("/api/compare")
async def compare(
    model_id_a: str = Form(...),
    model_id_b: str = Form(...),
    device: str = Form("cpu"),
    file: UploadFile = File(...),
) -> Dict[str, object]:
    if model_id_a not in MODEL_SPECS or model_id_b not in MODEL_SPECS:
        raise HTTPException(status_code=404, detail="Unknown model_id")

    device_obj = resolve_device(device)

    raw = await file.read()
    try:
        image = Image.open(io.BytesIO(raw)).convert("RGB")
    except Exception as exc:
        raise HTTPException(status_code=400, detail="Invalid image file") from exc

    return {"a": run_inference(model_id_a, image, device_obj), "b": run_inference(model_id_b, image, device_obj)}


@app.post("/api/compare-multi")
async def compare_multi(
    model_ids: List[str] = Form(...),
    device: str = Form("cpu"),
    file: UploadFile = File(...),
) -> Dict[str, object]:
    unique_model_ids: List[str] = list(dict.fromkeys(model_ids))
    if len(unique_model_ids) < 2:
        raise HTTPException(status_code=400, detail="Select at least 2 unique models")
    if len(unique_model_ids) > len(MODEL_SPECS):
        raise HTTPException(status_code=400, detail="Selected models exceed available models")

    for model_id in unique_model_ids:
        if model_id not in MODEL_SPECS:
            raise HTTPException(status_code=404, detail=f"Unknown model_id: {model_id}")

    device_obj = resolve_device(device)

    raw = await file.read()
    try:
        image = Image.open(io.BytesIO(raw)).convert("RGB")
    except Exception as exc:
        raise HTTPException(status_code=400, detail="Invalid image file") from exc

    return {
        "results": [run_inference(model_id, image, device_obj) for model_id in unique_model_ids]
    }
