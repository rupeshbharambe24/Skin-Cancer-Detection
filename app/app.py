import os
import html
from typing import Dict, Tuple

import numpy as np
import streamlit as st
import torch
from PIL import Image
from torchvision import transforms

from model_registry import count_parameters, get_model_specs


# -----------------------------
# Page
# -----------------------------
st.set_page_config(page_title="Lesion Classifier Studio", page_icon="🧫", layout="wide")


# -----------------------------
# Theme state (UI only)
# -----------------------------
if "theme" not in st.session_state:
    st.session_state["theme"] = "light"


def toggle_theme():
    st.session_state["theme"] = "dark" if st.session_state["theme"] == "light" else "light"


THEME = st.session_state["theme"]


# -----------------------------
# Theme CSS (NO f-string braces problems)
# -----------------------------
LIGHT_VARS = """
:root{
  --bg0:#f2f6f7;
  --bg1:#eef3f6;
  --panel:#ffffff;
  --text:#0f172a;
  --muted:#6b7280;
  --muted2:#94a3b8;
  --border:rgba(15,23,42,.08);
  --shadow:0 18px 40px rgba(15,23,42,.08);
  --shadow2:0 10px 22px rgba(15,23,42,.08);
  --accent:#14b8a6;
  --accent2:#22c55e;

  --chipbg:rgba(34,197,94,.14);
  --chipfg:#16a34a;

  --warnbg:rgba(239,68,68,.12);
  --warnfg:#b91c1c;

  --btnbg:linear-gradient(90deg,#14b8a6,#22c55e);
  --btnfg:#ffffff;

  --iconbg:rgba(20,184,166,.12);
  --iconfg:#0f766e;

  --soft:rgba(15,23,42,.03);
  --soft2:rgba(15,23,42,.04);

  --track:rgba(148,163,184,.28);
}
"""

DARK_VARS = """
:root{
  --bg0:#0b1220;
  --bg1:#0c172a;
  --panel:rgba(17,24,39,.62);
  --text:rgba(248,250,252,.95);
  --muted:rgba(226,232,240,.62);
  --muted2:rgba(226,232,240,.42);
  --border:rgba(148,163,184,.12);
  --shadow:0 22px 60px rgba(0,0,0,.55);
  --shadow2:0 14px 26px rgba(0,0,0,.45);
  --accent:#14b8a6;
  --accent2:#22c55e;

  --chipbg:rgba(34,197,94,.16);
  --chipfg:rgba(134,239,172,.92);

  --warnbg:rgba(239,68,68,.16);
  --warnfg:rgba(252,165,165,.92);

  --btnbg:linear-gradient(90deg,#22e0cf,#34d399);
  --btnfg:#07111f;

  --iconbg:rgba(20,184,166,.14);
  --iconfg:rgba(94,234,212,.92);

  --soft:rgba(255,255,255,.03);
  --soft2:rgba(255,255,255,.04);

  --track:rgba(148,163,184,.18);
}
"""

BASE_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700&family=Sora:wght@600;700&display=swap');

html, body, [class*="css"] {
  font-family: "Plus Jakarta Sans", system-ui, -apple-system, Segoe UI, Roboto, sans-serif;
}

.stApp{
  background:
    radial-gradient(1200px 700px at 22% 12%, rgba(20,184,166,.10), transparent 60%),
    radial-gradient(1100px 650px at 80% 18%, rgba(34,197,94,.10), transparent 65%),
    linear-gradient(180deg, var(--bg0), var(--bg1));
  color: var(--text);
}

#MainMenu { visibility: hidden; }
footer { visibility: hidden; }
header[data-testid="stHeader"] { display: none; }

.block-container{
  max-width: 1200px;
  padding-top: 1.1rem;
  padding-bottom: 1.6rem;
}

h1,h2,h3,h4{
  font-family: "Sora", system-ui, sans-serif;
  letter-spacing: -0.02em;
}

.topbar-wrap{
  padding: 14px 16px;
  border-radius: 18px;
  border: 1px solid var(--border);
  background: color-mix(in srgb, var(--panel) 92%, transparent);
  box-shadow: var(--shadow2);
  backdrop-filter: blur(10px);
}

.brand{
  display:flex;
  align-items:center;
  gap:12px;
}

.brand-ico{
  width: 44px;
  height: 44px;
  border-radius: 14px;
  background: var(--iconbg);
  color: var(--iconfg);
  display:flex;
  align-items:center;
  justify-content:center;
  font-weight: 900;
  font-size: 18px;
}

.brand-title{
  font-family:"Sora", system-ui, sans-serif;
  font-size: 22px;
  margin: 0;
  color: var(--text);
  line-height: 1.05;
}

.brand-sub{
  margin: 2px 0 0 0;
  font-size: 14px;
  color: var(--muted);
  font-weight: 500;
}

.status-pill{
  display:inline-flex;
  align-items:center;
  gap:10px;
  padding: 10px 14px;
  border-radius: 999px;
  font-weight: 800;
  border: 1px solid color-mix(in srgb, var(--accent2) 22%, var(--border));
  background: var(--chipbg);
  color: var(--chipfg);
}

.status-pill.danger{
  border-color: color-mix(in srgb, #ef4444 22%, var(--border));
  background: var(--warnbg);
  color: var(--warnfg);
}

.dot{
  width: 9px;
  height: 9px;
  border-radius: 999px;
  background: var(--accent2);
  box-shadow: 0 0 0 4px rgba(34,197,94,.12);
}

.danger .dot{
  background: #ef4444;
  box-shadow: 0 0 0 4px rgba(239,68,68,.12);
}

.card{
  border-radius: 20px;
  background: var(--panel);
  border: 1px solid var(--border);
  box-shadow: var(--shadow);
  padding: 18px 18px 16px 18px;
}

.card-head{
  display:flex;
  align-items:flex-start;
  gap:12px;
  margin-bottom: 14px;
}

.card-ico{
  width: 40px;
  height: 40px;
  border-radius: 14px;
  background: var(--iconbg);
  color: var(--iconfg);
  display:flex;
  align-items:center;
  justify-content:center;
  font-weight: 900;
  font-size: 18px;
}

.card-title{
  margin: 0;
  font-family:"Sora", system-ui, sans-serif;
  font-size: 18px;
  color: var(--text);
}

.card-sub{
  margin: 2px 0 0 0;
  color: var(--muted);
  font-size: 14px;
  font-weight: 500;
}

.field-label{
  color: var(--muted);
  font-size: 14px;
  font-weight: 700;
  margin: 10px 0 6px 0;
}

.sep{
  height:1px;
  background: var(--border);
  margin: 12px 0;
}

.toggle-row{
  display:flex;
  align-items:center;
  justify-content:space-between;
  padding: 12px 12px;
  border-radius: 14px;
  background: var(--soft);
  border: 1px solid var(--border);
}

.toggle-left{
  display:flex;
  align-items:center;
  gap:10px;
  font-weight: 800;
  color: var(--text);
}

.file-chip{
  display:inline-flex;
  align-items:center;
  padding: 5px 10px;
  border-radius: 999px;
  background: var(--iconbg);
  border: 1px solid color-mix(in srgb, var(--accent) 18%, var(--border));
  color: var(--iconfg);
  font-weight: 800;
  font-size: 13px;
  max-width: 100%;
  overflow:hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}

.kv{
  display:grid;
  grid-template-columns: 1fr auto;
  row-gap: 10px;
  column-gap: 10px;
  align-items:center;
  font-size: 15px;
}
.kv .k{ color: var(--muted); font-weight: 700; }
.kv .v{ color: var(--text); font-weight: 900; }

.dropwrap{ position: relative; }
.dropzone{
  border-radius: 18px;
  border: 2px dashed color-mix(in srgb, var(--accent) 28%, transparent);
  background: var(--soft);
  padding: 26px 14px;
  text-align:center;
}
.drop-ico{
  width: 58px;
  height: 58px;
  border-radius: 18px;
  background: var(--soft2);
  display:flex;
  align-items:center;
  justify-content:center;
  margin: 0 auto 12px auto;
  color: var(--muted2);
  font-weight: 900;
  font-size: 22px;
}
.drop-title{ margin:0; font-weight: 900; color: var(--text); }
.drop-sub{ margin:6px 0 0 0; color: var(--muted); font-size: 13px; font-weight: 700; }

.dropwrap div[data-testid="stFileUploader"]{
  position:absolute;
  inset:0;
  opacity:0;
}
.dropwrap div[data-testid="stFileUploader"] *{ cursor:pointer; }

.preview{
  margin-top: 14px;
  border-radius: 18px;
  overflow:hidden;
  border: 1px solid var(--border);
  background: var(--soft);
}

.empty{
  margin-top: 14px;
  border-radius: 18px;
  padding: 22px 12px;
  text-align:center;
  background: var(--soft);
  border: 1px solid var(--border);
}

.big{
  font-family:"Sora", system-ui, sans-serif;
  font-size: 34px;
  margin: 6px 0 0 0;
  color: var(--text);
}

.conf{
  display:inline-flex;
  align-items:center;
  padding: 8px 12px;
  border-radius: 999px;
  background: var(--chipbg);
  border: 1px solid color-mix(in srgb, var(--accent2) 22%, var(--border));
  color: var(--chipfg);
  font-weight: 900;
  font-size: 13px;
}

.prob-row{
  display:grid;
  grid-template-columns: 120px 1fr 70px;
  gap: 10px;
  align-items:center;
  padding: 10px 0;
}
.prob-name{ color: var(--muted); font-weight: 900; font-size: 14px; }
.prob-track{
  height: 10px;
  border-radius: 999px;
  background: var(--track);
  overflow:hidden;
  border: 1px solid var(--border);
}
.prob-fill{
  height:100%;
  border-radius: 999px;
  background: var(--btnbg);
}
.prob-val{ text-align:right; color: var(--text); font-weight: 900; font-size: 14px; }

.footer{
  margin-top: 22px;
  padding: 14px 0 8px 0;
  border-top: 1px solid var(--border);
  display:flex;
  align-items:center;
  justify-content:space-between;
  color: var(--muted);
  font-weight: 600;
  font-size: 14px;
}
.footer a{
  color: var(--muted);
  text-decoration:none;
  margin-left: 18px;
}
.footer a:hover{ text-decoration: underline; }

/* Buttons */
button[kind="primary"]{
  width:100% !important;
  border-radius: 14px !important;
  border: 0 !important;
  font-weight: 900 !important;
  padding: 12px 14px !important;
  background: var(--btnbg) !important;
  color: var(--btnfg) !important;
  box-shadow: 0 16px 28px rgba(20,184,166,.20) !important;
}
button[kind="primary"]:disabled{
  background: color-mix(in srgb, var(--text) 10%, transparent) !important;
  color: var(--muted2) !important;
  box-shadow: none !important;
}

/* Theme icon button (secondary) */
button[kind="secondary"]{
  width: 42px !important;
  border-radius: 14px !important;
  background: var(--soft2) !important;
  border: 1px solid var(--border) !important;
  box-shadow: none !important;
  font-size: 18px !important;
}

/* Select styling */
div[data-baseweb="select"] > div{ border-radius: 14px !important; }
</style>
"""

theme_css = LIGHT_VARS if THEME == "light" else DARK_VARS
st.markdown("<style>" + theme_css + BASE_CSS, unsafe_allow_html=True)


# -----------------------------
# Backend: specs (unchanged)
# -----------------------------
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

# UI defaults
if "selected_label" not in st.session_state:
    st.session_state["selected_label"] = spec_options[0][0]


# Compute status for header from current selection (updates on rerun)
_selected_key_for_header = [opt[1] for opt in spec_options if opt[0] == st.session_state["selected_label"]][0]
_selected_spec_for_header = spec_by_key[_selected_key_for_header]
_weights_exists_for_header = os.path.exists(_selected_spec_for_header.weights_path)
_status_class = "" if _weights_exists_for_header else "danger"
_status_text = "Model Ready" if _weights_exists_for_header else "Weights Missing"


# -----------------------------
# Top header bar (like your screenshot)
# -----------------------------
hl, hr = st.columns([0.74, 0.26], vertical_alignment="center")
with hl:
    st.markdown(
        """
<div class="topbar-wrap">
  <div class="brand">
    <div class="brand-ico">🧪</div>
    <div>
      <div class="brand-title">Lesion Classifier Studio</div>
      <div class="brand-sub">AI-Powered Dermatological Analysis</div>
    </div>
  </div>
</div>
""",
        unsafe_allow_html=True,
    )

with hr:
    st.markdown('<div class="topbar-wrap" style="display:flex;align-items:center;justify-content:flex-end;gap:10px;">', unsafe_allow_html=True)
    st.markdown(
        f'<div class="status-pill {_status_class}"><span class="dot"></span> {_status_text}</div>',
        unsafe_allow_html=True,
    )
    icon = "🌙" if THEME == "light" else "☀️"
    st.button(icon, key="theme_toggle", on_click=toggle_theme, type="secondary")
    st.markdown("</div>", unsafe_allow_html=True)

st.markdown("<div style='height:18px;'></div>", unsafe_allow_html=True)


# -----------------------------
# 3 columns (Model / Upload / Analysis) - single page
# -----------------------------
c1, c2, c3 = st.columns(3, gap="large")


# -----------------------------
# Card 1: Model & Runtime
# -----------------------------
with c1:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown(
        """
<div class="card-head">
  <div class="card-ico">⚙️</div>
  <div>
    <div class="card-title">Model &amp; Runtime</div>
    <div class="card-sub">Configure your classification model</div>
  </div>
</div>
""",
        unsafe_allow_html=True,
    )

    st.markdown('<div class="field-label">Model Architecture</div>', unsafe_allow_html=True)

    selected_label = st.selectbox(
        "Model Architecture",
        [opt[0] for opt in spec_options],
        index=[opt[0] for opt in spec_options].index(st.session_state["selected_label"]),
        key="selected_label",
        label_visibility="collapsed",
    )

    selected_key = [opt[1] for opt in spec_options if opt[0] == selected_label][0]
    selected_spec = spec_by_key[selected_key]

    # GPU row + toggle
    tl, tr = st.columns([0.72, 0.28], vertical_alignment="center")
    with tl:
        st.markdown(
            """
<div class="toggle-row">
  <div class="toggle-left">
    <span style="color:var(--accent);font-weight:900;">⚡</span>
    <span>GPU Acceleration</span>
  </div>
</div>
""",
            unsafe_allow_html=True,
        )
    with tr:
        use_gpu = st.toggle("GPU", value=True, label_visibility="collapsed", key="gpu_toggle")

    device = torch.device("cuda" if (use_gpu and torch.cuda.is_available()) else "cpu")

    weights_exists = os.path.exists(selected_spec.weights_path)
    weights_file = os.path.basename(selected_spec.weights_path)
    weights_full = selected_spec.weights_path

    st.markdown('<div class="sep"></div>', unsafe_allow_html=True)

    st.markdown(
        f"""
<div class="kv">
  <div class="k">Weights:</div>
  <div class="v"><span class="file-chip" title="{html.escape(weights_full)}">{html.escape(weights_file)}</span></div>

  <div class="k">Input Size</div>
  <div class="v">{selected_spec.input_size}×{selected_spec.input_size}</div>

  <div class="k">Classes</div>
  <div class="v">{len(selected_spec.class_names)}</div>

  <div class="k">Framework</div>
  <div class="v">PyTorch {torch.__version__}</div>
</div>
""",
        unsafe_allow_html=True,
    )

    st.markdown("<div style='height:14px;'></div>", unsafe_allow_html=True)

    if st.button("Reload Model", type="primary", use_container_width=True, key="reload_model"):
        load_model_cached.clear()
        st.success("Model cache cleared.")

    st.markdown("</div>", unsafe_allow_html=True)


# -----------------------------
# Card 2: Upload & Preview (with clickable dropzone overlay)
# -----------------------------
with c2:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown(
        """
<div class="card-head">
  <div class="card-ico">🖼️</div>
  <div>
    <div class="card-title">Upload &amp; Preview</div>
    <div class="card-sub">Drag and drop or select an image</div>
  </div>
</div>
""",
        unsafe_allow_html=True,
    )

    st.markdown('<div class="dropwrap">', unsafe_allow_html=True)
    st.markdown(
        """
<div class="dropzone">
  <div class="drop-ico">⤴</div>
  <p class="drop-title">Drop your image here</p>
  <p class="drop-sub">or click to browse (PNG, JPG, WEBP)</p>
</div>
""",
        unsafe_allow_html=True,
    )

    uploaded_file = st.file_uploader(
        "Upload image",
        type=["jpg", "jpeg", "png", "webp"],
        label_visibility="collapsed",
        key="uploader",
    )
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div style="height:8px"></div>', unsafe_allow_html=True)
    st.markdown('<div class="small-muted">• Supported formats: JPEG, PNG, WebP up to 10MB</div>', unsafe_allow_html=True)

    # Preview
    if uploaded_file is None:
        image = None
    else:
        image = Image.open(uploaded_file).convert("RGB")
        st.markdown('<div class="preview">', unsafe_allow_html=True)
        st.image(image, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)


# -----------------------------
# Session reset logic (UNCHANGED)
# -----------------------------
current_meta: Tuple[str, str] = (
    selected_key,
    uploaded_file.name if uploaded_file is not None else "",
)
if st.session_state["last_meta"] != current_meta:
    st.session_state["last_meta"] = current_meta
    st.session_state["last_result"] = None


# -----------------------------
# Card 3: Analysis & Results
# -----------------------------
with c3:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown(
        """
<div class="card-head">
  <div class="card-ico">📈</div>
  <div>
    <div class="card-title">Analysis &amp; Results</div>
    <div class="card-sub">AI-powered lesion classification</div>
  </div>
</div>
""",
        unsafe_allow_html=True,
    )

    analyze_disabled = (not weights_exists) or (image is None)

    analyze_clicked = st.button(
        "Analyze Image",
        type="primary",
        use_container_width=True,
        disabled=analyze_disabled,
        key="analyze",
    )

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
        conf = result["top_conf"] * 100.0
        st.markdown(
            f"""
<div style="display:flex;align-items:center;justify-content:space-between;gap:12px;margin-top:16px;">
  <div>
    <div class="big">{html.escape(str(result["top_label"]))}</div>
    <div class="small-muted" style="margin-top:6px;">Predicted class with highest probability</div>
  </div>
  <div class="conf">Confidence&nbsp;&nbsp;{conf:.2f}%</div>
</div>
<div class="sep"></div>
<div class="small-muted" style="font-weight:900;margin-bottom:6px;">Class probabilities</div>
""",
            unsafe_allow_html=True,
        )

        for name, prob in zip(result["class_names"], result["probs"]):
            w = max(0.0, min(100.0, float(prob) * 100.0))
            st.markdown(
                f"""
<div class="prob-row">
  <div class="prob-name">{html.escape(str(name))}</div>
  <div class="prob-track"><div class="prob-fill" style="width:{w:.2f}%"></div></div>
  <div class="prob-val">{w:.2f}%</div>
</div>
""",
                unsafe_allow_html=True,
            )
    else:
        if not weights_exists:
            st.markdown(
                """
<div class="empty">
  <div style="font-size:46px;opacity:.6;margin-bottom:10px;">⚠️</div>
  <div class="small-muted" style="font-weight:900;">Weights missing — update <code>model_registry.py</code></div>
</div>
""",
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                """
<div class="empty">
  <div style="font-size:52px;opacity:.55;margin-bottom:10px;">〰️</div>
  <div class="small-muted" style="font-weight:900;">Upload an image to begin analysis</div>
</div>
""",
                unsafe_allow_html=True,
            )

    st.markdown("</div>", unsafe_allow_html=True)


# -----------------------------
# Footer (like screenshot)
# -----------------------------
st.markdown(
    """
<div class="footer">
  <div>© 2026 Lesion Classifier Studio. For demonstration purposes only.</div>
  <div>
    <a href="#">Documentation</a>
    <a href="#">API Reference</a>
    <a href="#">Support</a>
  </div>
</div>
""",
    unsafe_allow_html=True,
)
