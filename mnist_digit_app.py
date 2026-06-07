import streamlit as st
import numpy as np
from PIL import Image, ImageOps, ImageFilter
import io
import cv2
from streamlit_drawable_canvas import st_canvas

st.set_page_config(page_title="MNIST Digit Converter", layout="centered")

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
</style>
""", unsafe_allow_html=True)

# ─── IMAGE PREPROCESSING FUNCTIONS ───────────────────────────────────────────

def convert_to_grayscale(img_array):
    """Convert RGBA canvas image to grayscale."""
    img = Image.fromarray(img_array.astype(np.uint8))
    return np.array(img.convert('L'))

def crop_whitespace(gray_array):
    """Crop unnecessary whitespace around the digit using contours."""
    # Find non-zero pixels (the drawn digit)
    coords = cv2.findNonZero(gray_array)
    if coords is None:
        return gray_array
    x, y, w, h = cv2.boundingRect(coords)
    # Add small padding around digit
    padding = 10
    x = max(0, x - padding)
    y = max(0, y - padding)
    w = min(gray_array.shape[1] - x, w + 2 * padding)
    h = min(gray_array.shape[0] - y, h + 2 * padding)
    return gray_array[y:y+h, x:x+w]

def center_digit(gray_array, target_size=20):
    """Center the digit in a square canvas."""
    h, w = gray_array.shape
    # Resize to fit within target_size x target_size keeping aspect ratio
    if h > w:
        new_h = target_size
        new_w = max(1, int(w * target_size / h))
    else:
        new_w = target_size
        new_h = max(1, int(h * target_size / w))
    
    resized = cv2.resize(gray_array, (new_w, new_h), interpolation=cv2.INTER_AREA)
    
    # Place on 28x28 black canvas centered
    canvas = np.zeros((28, 28), dtype=np.uint8)
    y_offset = (28 - new_h) // 2
    x_offset = (28 - new_w) // 2
    canvas[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized
    return canvas

def apply_antialiasing(gray_array):
    """Apply Gaussian blur for anti-aliasing effect."""
    return cv2.GaussianBlur(gray_array, (3, 3), 0)

def normalize_pixels(gray_array):
    """Normalize pixel values to 0-255 range."""
    if gray_array.max() == 0:
        return gray_array
    normalized = (gray_array.astype(float) / gray_array.max() * 255).astype(np.uint8)
    return normalized

def preprocess_to_mnist(img_array):
    """
    Full MNIST preprocessing pipeline:
    1. Grayscale conversion
    2. Crop whitespace
    3. Center digit
    4. Anti-aliasing
    5. Normalize to 0-255
    """
    gray = convert_to_grayscale(img_array)
    cropped = crop_whitespace(gray)
    centered = center_digit(cropped)
    antialiased = apply_antialiasing(centered)
    normalized = normalize_pixels(antialiased)
    return normalized

# ─── UI ───────────────────────────────────────────────────────────────────────

st.markdown("<h1>MNIST Digit Converter</h1>", unsafe_allow_html=True)
st.markdown('<p class="subtitle">Draw → Preprocess → 28×28 px · Grayscale · MNIST Standard</p>', unsafe_allow_html=True)

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
        key="canvas",
        display_toolbar=True,
    )
    
    # Process button
    process_clicked = st.button("⚡ Process Digit")

with col2:
    st.markdown('<p class="section-header">MNIST Output (28×28 px)</p>', unsafe_allow_html=True)

    if canvas_result.image_data is not None and process_clicked:
        img_array = canvas_result.image_data.astype(np.uint8)

        if img_array.sum() > 0:
            # Run full preprocessing pipeline
            processed = preprocess_to_mnist(img_array)

            # Display enlarged
            img_display = Image.fromarray(processed).resize((280, 280), Image.NEAREST)
            st.image(img_display, caption="28×28 MNIST output", use_column_width=True)

            # Metrics
            non_zero = int(np.count_nonzero(processed))
            max_val = int(processed.max())
            min_val = int(processed.min())
            mean_val = round(float(processed[processed > 0].mean()), 1) if non_zero > 0 else 0

            m1, m2, m3 = st.columns(3)
            with m1:
                st.markdown(f'<div class="metric-box"><div class="metric-label">Non-zero px</div><div class="metric-value">{non_zero}</div></div>', unsafe_allow_html=True)
            with m2:
                st.markdown(f'<div class="metric-box"><div class="metric-label">Max value</div><div class="metric-value">{max_val}</div></div>', unsafe_allow_html=True)
            with m3:
                st.markdown(f'<div class="metric-box"><div class="metric-label">Mean (ink)</div><div class="metric-value">{mean_val}</div></div>', unsafe_allow_html=True)

            # Image dimensions
            st.markdown(f'<p class="section-header">Dimensions: {processed.shape[0]} × {processed.shape[1]} px · Range: {min_val}–{max_val}</p>', unsafe_allow_html=True)

            # Save processed_digit.png
            output_img = Image.fromarray(processed)
            output_img.save("processed_digit.png")

            # Download buttons
            st.markdown('<p class="section-header">Export</p>', unsafe_allow_html=True)
            dl1, dl2 = st.columns(2)
            with dl1:
                buf = io.BytesIO()
                output_img.save(buf, format="PNG")
                st.download_button("⬇ PNG", buf.getvalue(), "processed_digit.png", "image/png")
            with dl2:
                csv_data = '\n'.join([','.join(map(str, row)) for row in processed])
                st.download_button("⬇ CSV", csv_data, "mnist_digit.csv", "text/csv")

            # Raw pixel grid
            st.markdown('<p class="section-header">Raw Pixel Values (28×28)</p>', unsafe_allow_html=True)
            st.dataframe(processed, height=200)

        else:
            st.warning("Canvas is empty — please draw a digit first!")

    elif not process_clicked:
        st.markdown("""
        <div style="height:280px; display:flex; align-items:center; justify-content:center; 
                    border: 1px dashed #222; border-radius:8px; color:#333; 
                    font-family:'Space Mono',monospace; font-size:0.8rem; text-align:center;">
            Draw a digit, then click<br>⚡ Process Digit
        </div>
        """, unsafe_allow_html=True)

# ─── PREPROCESSING PIPELINE EXPLANATION ──────────────────────────────────────
st.markdown("---")
st.markdown('<p class="section-header">Preprocessing Pipeline</p>', unsafe_allow_html=True)

steps = [
    "1. Grayscale Conversion — RGBA canvas → single channel (0–255)",
    "2. Crop Whitespace — OpenCV contour detection removes empty borders",
    "3. Center Digit — digit scaled to 20×20 and centered in 28×28 canvas",
    "4. Anti-aliasing — Gaussian blur smooths jagged edges",
    "5. Normalize — pixel values scaled to full 0–255 range",
]
for step in steps:
    st.markdown(f'<div class="pipeline-step">{step}</div>', unsafe_allow_html=True)

st.markdown("""
<p style="font-family:'Space Mono',monospace; color:#333; font-size:0.7rem; text-align:center; margin-top:1rem;">
MNIST Standard · 28×28px · Grayscale · Anti-aliased · Pixel values 0–255 · Ready for Neural Network input
</p>
""", unsafe_allow_html=True)
