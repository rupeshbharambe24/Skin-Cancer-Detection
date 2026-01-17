import os
from typing import Dict

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
@import url('https://fonts.googleapis.com/css2?family=Manrope:wght@300;500;700&family=Space+Grotesk:wght@500;700&display=swap');

.stApp {
  background: linear-gradient(135deg, #f7fbff 0%, #f2f7ff 40%, #f7faf5 100%);
  color: #0f172a;
}

h1, h2, h3, h4 {
  font-family: 'Space Grotesk', sans-serif;
  letter-spacing: -0.02em;
}

p, div, span, label {
  font-family: 'Manrope', sans-serif;
}

.hero {
  background: linear-gradient(120deg, #ffffff 0%, #f5f8ff 55%, #f3fbf2 100%);
  border: 1px solid #e7edf7;
  border-radius: 18px;
  padding: 24px 26px;
  box-shadow: 0 12px 30px rgba(15, 23, 42, 0.06);
  margin-bottom: 18px;
}

.eyebrow {
  text-transform: uppercase;
  font-size: 12px;
  letter-spacing: 0.18em;
  color: #64748b;
  font-weight: 600;
}

.card {
  background: #ffffff;
  border: 1px solid #e7edf7;
  border-radius: 16px;
  padding: 18px;
  box-shadow: 0 10px 24px rgba(15, 23, 42, 0.05);
}

.badge {
  display: inline-block;
  padding: 6px 10px;
  border-radius: 999px;
  background: #e8f0ff;
  color: #1d4ed8;
  font-size: 12px;
  font-weight: 600;
}

.metric {
  font-size: 28px;
  font-weight: 700;
}

.stButton>button {
  background: #0f172a;
  color: white;
  border-radius: 10px;
  padding: 10px 18px;
  border: none;
}

.stButton>button:hover {
  background: #1f2937;
  color: #ffffff;
}

.image-frame {
  border-radius: 14px;
  border: 1px dashed #d7e0ee;
  padding: 14px;
  background: #f8fafc;
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
  <p>Pick a model, upload a dermoscopy image, and get a prediction instantly. Swap models or weights anytime.</p>
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
    spec_options.append((label, spec.key, params_m))
    spec_by_key[spec.key] = spec

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
    st.markdown(
        f"<div class=\"badge\">Weights: {'Found' if weights_exists else 'Missing'}</div>",
        unsafe_allow_html=True,
    )

    st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)
    st.write(f"Weights file: `{selected_spec.weights_path}`")
    st.write(f"Input size: `{selected_spec.input_size}x{selected_spec.input_size}`")
    st.write(f"Device: `{device.type}`")
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div style='height: 14px;'></div>", unsafe_allow_html=True)
    st.markdown("<div class=\"card\">", unsafe_allow_html=True)
    st.markdown("<div class=\"eyebrow\">Upload</div>", unsafe_allow_html=True)
    uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])
    st.markdown("</div>", unsafe_allow_html=True)


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


with right:
    st.markdown("<div class=\"card\">", unsafe_allow_html=True)
    st.markdown("<div class=\"eyebrow\">Preview</div>", unsafe_allow_html=True)

    if uploaded_file is None:
        st.markdown("<div class='image-frame'>Upload an image to preview and predict.</div>", unsafe_allow_html=True)
    else:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, use_container_width=True, caption="Uploaded image")

    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div style='height: 14px;'></div>", unsafe_allow_html=True)
    st.markdown("<div class=\"card\">", unsafe_allow_html=True)
    st.markdown("<div class=\"eyebrow\">Prediction</div>", unsafe_allow_html=True)

    if not weights_exists:
        st.error("Weights file not found. Update the path in model_registry.py.")
    elif uploaded_file is None:
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

        st.markdown(f"<div class=\"metric\">{top_label}</div>", unsafe_allow_html=True)
        st.write(f"Confidence: `{top_conf * 100:.2f}%`")

        st.write("Class probabilities:")
        for name, prob in zip(class_names, probs):
            st.write(f"{name}: {prob * 100:.2f}%")
            st.progress(float(prob))

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
