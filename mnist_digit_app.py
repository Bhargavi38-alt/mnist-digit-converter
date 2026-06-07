import streamlit as st
import numpy as np
from PIL import Image
import io
import cv2
import pandas as pd
from streamlit_drawable_canvas import st_canvas

st.set_page_config(page_title="MNIST Digit Collector", layout="centered")

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Inter:wght@300;400;600&display=swap');
    body { background-color: #0a0a0a; }
    .main { background-color: #0a0a0a; }
    h1 {
        font-family: 'Space Mono', monospace;
        color: #00ff88;
        font-size: 1.8rem;
        letter-spacing: -1px;
        margin-bottom: 0;
    }
    .subtitle {
        font-family: 'Inter', sans-serif;
        color: #666;
        font-size: 0.85rem;
        margin-bottom: 2rem;
        letter-spacing: 2px;
        text-transform: uppercase;
    }
    .metric-box {
        background: #111;
        border: 1px solid #222;
        border-radius: 8px;
        padding: 1rem;
        text-align: center;
        font-family: 'Space Mono', monospace;
    }
    .metric-label { color: #555; font-size: 0.7rem; text-transform: uppercase; letter-spacing: 2px; }
    .metric-value { color: #00ff88; font-size: 1.4rem; font-weight: 700; }
    .section-header {
        font-family: 'Space Mono', monospace;
        color: #444;
        font-size: 0.7rem;
        text-transform: uppercase;
        letter-spacing: 3px;
        margin: 1.5rem 0 0.5rem 0;
    }
    .stButton > button {
        font-family: 'Space Mono', monospace;
        background: #00ff88;
        color: #000;
        border: none;
        border-radius: 4px;
        font-weight: 700;
        letter-spacing: 1px;
        width: 100%;
    }
    .stButton > button:hover { background: #00cc6a; color: #000; }
    .pipeline-step {
        background: #111;
        border-left: 3px solid #00ff88;
        padding: 0.5rem 1rem;
        margin: 0.3rem 0;
        font-family: 'Space Mono', monospace;
        font-size: 0.75rem;
        color: #888;
    }
    .progress-box {
        background: #111;
        border: 1px solid #222;
        border-radius: 8px;
        padding: 1rem;
        font-family: 'Space Mono', monospace;
        margin-bottom: 1rem;
    }
    .digit-collected { color: #00ff88; }
    .digit-pending { color: #333; }
</style>
""", unsafe_allow_html=True)

# ─── PREPROCESSING FUNCTIONS ──────────────────────────────────────────────────

def convert_to_grayscale(img_array):
    img = Image.fromarray(img_array.astype(np.uint8))
    return np.array(img.convert('L'))

def crop_whitespace(gray_array):
    coords = cv2.findNonZero(gray_array)
    if coords is None:
        return gray_array
    x, y, w, h = cv2.boundingRect(coords)
    padding = 10
    x = max(0, x - padding)
    y = max(0, y - padding)
    w = min(gray_array.shape[1] - x, w + 2 * padding)
    h = min(gray_array.shape[0] - y, h + 2 * padding)
    return gray_array[y:y+h, x:x+w]

def center_digit(gray_array, target_size=20):
    h, w = gray_array.shape
    if h > w:
        new_h = target_size
        new_w = max(1, int(w * target_size / h))
    else:
        new_w = target_size
        new_h = max(1, int(h * target_size / w))
    resized = cv2.resize(gray_array, (new_w, new_h), interpolation=cv2.INTER_AREA)
    canvas = np.zeros((28, 28), dtype=np.uint8)
    y_offset = (28 - new_h) // 2
    x_offset = (28 - new_w) // 2
    canvas[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized
    return canvas

def apply_antialiasing(gray_array):
    return cv2.GaussianBlur(gray_array, (3, 3), 0)

def normalize_pixels(gray_array):
    if gray_array.max() == 0:
        return gray_array
    return (gray_array.astype(float) / gray_array.max() * 255).astype(np.uint8)

def preprocess_to_mnist(img_array):
    gray = convert_to_grayscale(img_array)
    cropped = crop_whitespace(gray)
    centered = center_digit(cropped)
    antialiased = apply_antialiasing(centered)
    normalized = normalize_pixels(antialiased)
    return normalized

# ─── SESSION STATE ────────────────────────────────────────────────────────────

if 'collected' not in st.session_state:
    st.session_state.collected = {}  # {digit: pixel_array}

# ─── UI ───────────────────────────────────────────────────────────────────────

st.markdown("<h1>MNIST Digit Collector</h1>", unsafe_allow_html=True)
st.markdown('<p class="subtitle">Draw all 10 digits → Export class dataset</p>', unsafe_allow_html=True)

# User name input
user_name = st.text_input("Your name (as it will appear in dataset)", placeholder="e.g. Bhargavi")

if not user_name:
    st.warning("Please enter your name above to begin.")
    st.stop()

# Progress tracker
collected_digits = list(st.session_state.collected.keys())
st.markdown('<p class="section-header">Collection Progress</p>', unsafe_allow_html=True)
progress_html = '<div class="progress-box">Digits collected: '
for d in range(10):
    if d in collected_digits:
        progress_html += f'<span class="digit-collected">✅{d} </span>'
    else:
        progress_html += f'<span class="digit-pending">⬜{d} </span>'
progress_html += f'<br><br>{len(collected_digits)}/10 complete</div>'
st.markdown(progress_html, unsafe_allow_html=True)

# Digit selector
current_digit = st.selectbox("Select digit you are about to draw", list(range(10)))

col1, col2 = st.columns([1.2, 1])

with col1:
    st.markdown('<p class="section-header">Draw your digit</p>', unsafe_allow_html=True)
    stroke_width = st.slider("Brush size", 8, 30, 18, label_visibility="collapsed")

    canvas_result = st_canvas(
        fill_color="rgba(255, 255, 255, 0)",
        stroke_width=stroke_width,
        stroke_color="#FFFFFF",
        background_color="#000000",
        height=280,
        width=280,
        drawing_mode="freedraw",
        key=f"canvas_{current_digit}",
        display_toolbar=True,
    )

    process_clicked = st.button(f"⚡ Add Digit {current_digit} to Collection")

with col2:
    st.markdown('<p class="section-header">MNIST Output (28×28 px)</p>', unsafe_allow_html=True)

    if canvas_result.image_data is not None and process_clicked:
        img_array = canvas_result.image_data.astype(np.uint8)

        if img_array.sum() > 0:
            processed = preprocess_to_mnist(img_array)

            # Store in session state
            st.session_state.collected[current_digit] = processed.flatten()

            # Display
            img_display = Image.fromarray(processed).resize((280, 280), Image.NEAREST)
            st.image(img_display, caption=f"Digit {current_digit} — 28×28 MNIST", use_column_width=True)

            non_zero = int(np.count_nonzero(processed))
            max_val = int(processed.max())
            mean_val = round(float(processed[processed > 0].mean()), 1) if non_zero > 0 else 0

            m1, m2, m3 = st.columns(3)
            with m1:
                st.markdown(f'<div class="metric-box"><div class="metric-label">Non-zero px</div><div class="metric-value">{non_zero}</div></div>', unsafe_allow_html=True)
            with m2:
                st.markdown(f'<div class="metric-box"><div class="metric-label">Max value</div><div class="metric-value">{max_val}</div></div>', unsafe_allow_html=True)
            with m3:
                st.markdown(f'<div class="metric-box"><div class="metric-label">Mean (ink)</div><div class="metric-value">{mean_val}</div></div>', unsafe_allow_html=True)

            st.success(f"✅ Digit {current_digit} added! {len(st.session_state.collected)}/10 collected.")
        else:
            st.warning("Canvas is empty — please draw a digit first!")

    elif current_digit in st.session_state.collected:
        stored = st.session_state.collected[current_digit].reshape(28, 28).astype(np.uint8)
        img_display = Image.fromarray(stored).resize((280, 280), Image.NEAREST)
        st.image(img_display, caption=f"Digit {current_digit} already collected ✅", use_column_width=True)
    else:
        st.markdown("""
        <div style="height:280px; display:flex; align-items:center; justify-content:center; 
                    border: 1px dashed #222; border-radius:8px; color:#333; 
                    font-family:'Space Mono',monospace; font-size:0.8rem; text-align:center;">
            Draw digit, then click<br>⚡ Add to Collection
        </div>
        """, unsafe_allow_html=True)

# ─── EXPORT COMBINED CSV ──────────────────────────────────────────────────────

st.markdown("---")
st.markdown('<p class="section-header">Export Dataset</p>', unsafe_allow_html=True)

if len(st.session_state.collected) == 10:
    st.success("🎉 All 10 digits collected! Ready to export.")

    rows = []
    for digit in range(10):
        pixel_values = st.session_state.collected[digit]
        row = {'user': user_name, 'label': digit}
        for i, val in enumerate(pixel_values):
            row[f'pixel_{i}'] = val
        rows.append(row)

    df = pd.DataFrame(rows)
    csv_data = df.to_csv(index=False)

    st.download_button(
        "⬇ Download Combined CSV",
        csv_data,
        f"custom_mnist_{user_name.replace(' ', '_')}.csv",
        "text/csv"
    )
else:
    remaining = 10 - len(st.session_state.collected)
    st.info(f"Draw {remaining} more digit(s) to unlock export.")

# ─── PIPELINE ─────────────────────────────────────────────────────────────────
st.markdown('<p class="section-header">Preprocessing Pipeline</p>', unsafe_allow_html=True)
steps = [
    "1. Grayscale Conversion — RGBA canvas → single channel (0–255)",
    "2. Crop Whitespace — OpenCV contour detection removes empty borders",
    "3. Center Digit — digit scaled to 20×20 and centered in 28×28 canvas",
    "4. Anti-aliasing — Gaussian blur smooths jagged edges",
    "5. Normalize — pixel values scaled to full 0–255 range",
    "6. Flatten — 28×28 image → 784 pixel columns for dataset",
]
for step in steps:
    st.markdown(f'<div class="pipeline-step">{step}</div>', unsafe_allow_html=True)

st.markdown("""
<p style="font-family:'Space Mono',monospace; color:#333; font-size:0.7rem; text-align:center; margin-top:1rem;">
MNIST Standard · 28×28px · Grayscale · user + label + 784 pixels · Ready for Neural Network
</p>
""", unsafe_allow_html=True)
