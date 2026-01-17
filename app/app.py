import os
from typing import Dict, Tuple

import numpy as np
import streamlit as st
import torch
from PIL import Image
from torchvision import transforms

from model_registry import count_parameters, get_model_specs


st.set_page_config(page_title="Skin Lesion Classifier", page_icon="S", layout="wide")

st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700&family=Sora:wght@500;700&display=swap');

:root {
  --bg: #f5f7fb;
  --panel: #ffffff;
  --border: #e4eaf3;
  --muted: #64748b;
  --text: #0f172a;
  --accent: #0ea5a4;
  --accent-2: #2563eb;
  --soft: #eef2ff;
}

.stApp {
  background: var(--bg);
  color: var(--text);
}

.stApp:before {
  content: "";
  position: fixed;
  inset: 0;
  background:
    radial-gradient(circle at 15% 20%, rgba(14, 165, 164, 0.08), transparent 35%),
    radial-gradient(circle at 85% 5%, rgba(37, 99, 235, 0.12), transparent 30%),
    radial-gradient(circle at 80% 80%, rgba(16, 185, 129, 0.10), transparent 40%);
  pointer-events: none;
  z-index: -1;
}

h1, h2, h3, h4 {
  font-family: 'Sora', sans-serif;
  letter-spacing: -0.02em;
  color: var(--text);
}

p, div, span, label {
  font-family: 'Plus Jakarta Sans', sans-serif;
}

.block-container {
  padding-top: 2.2rem;
  padding-bottom: 3rem;
  max-width: 1200px;
}

.hero {
  background: linear-gradient(120deg, #ffffff 0%, #f3f7ff 55%, #f1fbfa 100%);
  border: 1px solid var(--border);
  border-radius: 20px;
  padding: 26px 28px;
  box-shadow: 0 18px 40px rgba(15, 23, 42, 0.08);
  margin-bottom: 20px;
}

.eyebrow {
  text-transform: uppercase;
  font-size: 11px;
  letter-spacing: 0.22em;
  color: var(--muted);
  font-weight: 600;
}

.card {
  background: var(--panel);
  border: 1px solid var(--border);
  border-radius: 18px;
  padding: 18px;
  box-shadow: 0 12px 28px rgba(15, 23, 42, 0.06);
}

.chip {
  display: inline-flex;
  align-items: center;
  gap: 6px;
  padding: 6px 10px;
  border-radius: 999px;
  font-size: 12px;
  font-weight: 600;
  border: 1px solid transparent;
}

.chip.ok {
  background: #ecfdf5;
  color: #047857;
  border-color: #bbf7d0;
}

.chip.warn {
  background: #fff7ed;
  color: #9a3412;
  border-color: #fed7aa;
}

.metric {
  font-size: 30px;
  font-weight: 700;
  margin-top: 6px;
}

.stButton>button {
  background: linear-gradient(120deg, #0ea5a4, #2563eb);
  color: white;
  border-radius: 12px;
  padding: 10px 18px;
  border: none;
  font-weight: 600;
  box-shadow: 0 10px 20px rgba(37, 99, 235, 0.2);
}

.stButton>button:hover {
  background: linear-gradient(120deg, #0f766e, #1d4ed8);
  color: #ffffff;
}

.stButton>button:disabled {
  background: #cbd5f5;
  color: #475569;
  box-shadow: none;
}

div[data-testid="stFileUploader"] {
  border: 1px dashed #cbd5f5;
  border-radius: 14px;
  padding: 8px;
  background: #f8fafc;
}

div[data-baseweb="select"] > div {
  border-radius: 12px;
}

.image-frame {
  border-radius: 14px;
  border: 1px dashed #d7e0ee;
  padding: 12px;
  background: #f8fafc;
  color: var(--muted);
}

.small-note {
  color: var(--muted);
  font-size: 13px;
}

@media (max-width: 900px) {
  .hero {
    padding: 18px;
  }
}
</style>
""",
    unsafe_allow_html=True,
)

st.markdown(
    """
<div class="hero">
  <div class="eyebrow">Skin Lesion AI</div>
  <h1>Lesion Classifier Studio</h1>
  <p>Choose a model, upload a dermoscopy image, and run analysis on demand. Swap architectures or weights anytime.</p>
</div>
""",
    unsafe_allow_html=True,
)

specs = get_model_specs()

spec_options = []
spec_by_key: Dict[str, object] = {}
for spec in specs:
    model = spec.builder()
    params = count_parameters(model)
    params_m = params / 1_000_000
    label = f"{spec.name}  |  {params_m:.2f}M params"
    spec_options.append((label, spec.key))
    spec_by_key[spec.key] = spec


@st.cache_resource
def load_model_cached(spec_key: str, device_type: str):
    spec = spec_by_key[spec_key]
    model = spec.builder()
    state = torch.load(spec.weights_path, map_location="cpu")
    model.load_state_dict(state)
    model.to(device_type)
    model.eval()
    return model


def build_transform(spec):
    return transforms.Compose(
        [
            transforms.Resize((spec.input_size, spec.input_size)),
            transforms.ToTensor(),
            transforms.Normalize(spec.norm_mean, spec.norm_std),
        ]
    )


if "last_result" not in st.session_state:
    st.session_state["last_result"] = None
if "last_meta" not in st.session_state:
    st.session_state["last_meta"] = (None, None)

left, right = st.columns([0.38, 0.62], gap="large")

with left:
    st.markdown("<div class=\"card\">", unsafe_allow_html=True)
    st.markdown("<div class=\"eyebrow\">Model</div>", unsafe_allow_html=True)
    selected_label = st.selectbox(
        "Select model",
        [opt[0] for opt in spec_options],
        index=0,
    )
    selected_key = [opt[1] for opt in spec_options if opt[0] == selected_label][0]
    selected_spec = spec_by_key[selected_key]

    use_gpu = st.toggle("Use GPU if available", value=True)
    device = torch.device("cuda" if (use_gpu and torch.cuda.is_available()) else "cpu")

    weights_exists = os.path.exists(selected_spec.weights_path)
    chip_class = "ok" if weights_exists else "warn"
    chip_label = "Weights found" if weights_exists else "Weights missing"
    st.markdown(
        f"<div class=\"chip {chip_class}\">{chip_label}</div>",
        unsafe_allow_html=True,
    )

    st.markdown("<div style='height: 8px;'></div>", unsafe_allow_html=True)
    st.write(f"Weights file: `{selected_spec.weights_path}`")
    st.write(f"Input size: `{selected_spec.input_size}x{selected_spec.input_size}`")
    st.write(f"Device: `{device.type}`")
    st.markdown("<div class='small-note'>Tip: update weights and normalization in `model_registry.py`.</div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div style='height: 14px;'></div>", unsafe_allow_html=True)
    st.markdown("<div class=\"card\">", unsafe_allow_html=True)
    st.markdown("<div class=\"eyebrow\">Upload</div>", unsafe_allow_html=True)
    uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])
    st.caption("Supported: JPG, PNG. Best results with dermoscopy images.")

    if st.button("Reload model weights", use_container_width=True):
        load_model_cached.clear()
        st.success("Model cache cleared.")

    st.markdown("</div>", unsafe_allow_html=True)


current_meta: Tuple[str, str] = (
    selected_key,
    uploaded_file.name if uploaded_file is not None else "",
)
if st.session_state["last_meta"] != current_meta:
    st.session_state["last_meta"] = current_meta
    st.session_state["last_result"] = None


with right:
    st.markdown("<div class=\"card\">", unsafe_allow_html=True)
    st.markdown("<div class=\"eyebrow\">Preview</div>", unsafe_allow_html=True)

    if uploaded_file is None:
        st.markdown("<div class='image-frame'>Upload an image to preview and analyze.</div>", unsafe_allow_html=True)
        image = None
    else:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, width=360, caption="Uploaded image")

    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div style='height: 14px;'></div>", unsafe_allow_html=True)
    st.markdown("<div class=\"card\">", unsafe_allow_html=True)
    st.markdown("<div class=\"eyebrow\">Analysis</div>", unsafe_allow_html=True)

    analyze_disabled = (not weights_exists) or (image is None)
    analyze_clicked = st.button("Analyze image", use_container_width=True, disabled=analyze_disabled)

    if analyze_clicked:
        if not weights_exists:
            st.error("Weights file not found. Update the path in model_registry.py.")
        elif image is None:
            st.info("Upload an image to run inference.")
        else:
            model = load_model_cached(selected_key, device.type)
            transform = build_transform(selected_spec)
            tensor = transform(image).unsqueeze(0).to(device)
            with torch.no_grad():
                logits = model(tensor)
                probs = torch.softmax(logits, dim=1).cpu().numpy().flatten()

            class_names = selected_spec.class_names
            top_idx = int(np.argmax(probs))
            top_label = class_names[top_idx]
            top_conf = float(probs[top_idx])

            st.session_state["last_result"] = {
                "class_names": class_names,
                "probs": probs,
                "top_label": top_label,
                "top_conf": top_conf,
            }

    result = st.session_state.get("last_result")
    if result:
        st.markdown(f"<div class=\"metric\">{result['top_label']}</div>", unsafe_allow_html=True)
        st.write(f"Confidence: `{result['top_conf'] * 100:.2f}%`")
        st.write("Class probabilities:")
        for name, prob in zip(result["class_names"], result["probs"]):
            st.write(f"{name}: {prob * 100:.2f}%")
            st.progress(float(prob))
    elif not weights_exists:
        st.error("Weights file not found. Update the path in model_registry.py.")
    elif image is None:
        st.info("Upload an image to run inference.")

    st.markdown("</div>", unsafe_allow_html=True)

st.markdown(
    """
<div style="height: 16px"></div>
<div class="card">
  <div class="eyebrow">How to extend</div>
  <p>Add a new entry in <code>model_registry.py</code> with your builder and weights path. The UI will pick it up automatically.</p>
</div>
""",
    unsafe_allow_html=True,
)
