import streamlit as st
import numpy as np
from PIL import Image, ImageOps, ImageFilter
import io
import base64
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
    .metric-label {
        color: #555;
        font-size: 0.7rem;
        text-transform: uppercase;
        letter-spacing: 2px;
    }
    .metric-value {
        color: #00ff88;
        font-size: 1.4rem;
        font-weight: 700;
    }
    .section-header {
        font-family: 'Space Mono', monospace;
        color: #444;
        font-size: 0.7rem;
        text-transform: uppercase;
        letter-spacing: 3px;
        margin: 1.5rem 0 0.5rem 0;
    }
    .pixel-grid-container {
        background: #0a0a0a;
        border: 1px solid #1a1a1a;
        border-radius: 8px;
        padding: 1rem;
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
    .stButton > button:hover {
        background: #00cc6a;
        color: #000;
    }
</style>
""", unsafe_allow_html=True)

st.markdown("<h1>MNIST Digit Converter</h1>", unsafe_allow_html=True)
st.markdown('<p class="subtitle">Draw → 28×28 px · Grayscale · MNIST Standard</p>', unsafe_allow_html=True)

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

with col2:
    st.markdown('<p class="section-header">MNIST Output (28×28 px)</p>', unsafe_allow_html=True)
    
    if canvas_result.image_data is not None:
        img_array = canvas_result.image_data.astype(np.uint8)
        
        if img_array.sum() > 0:
            # Convert to PIL
            img = Image.fromarray(img_array)
            
            # Convert to grayscale
            img_gray = img.convert('L')
            
            # Resize to 28x28 with anti-aliasing
            img_resized = img_gray.resize((28, 28), Image.LANCZOS)
            
            # Normalize
            img_array_28 = np.array(img_resized)
            
            # Display 28x28 enlarged
            img_display = img_resized.resize((280, 280), Image.NEAREST)
            st.image(img_display, caption="28×28 pixel output", use_column_width=True)
            
            # Metrics
            non_zero = np.count_nonzero(img_array_28)
            max_val = int(img_array_28.max())
            mean_val = round(float(img_array_28[img_array_28 > 0].mean()), 1) if non_zero > 0 else 0
            
            m1, m2, m3 = st.columns(3)
            with m1:
                st.markdown(f'<div class="metric-box"><div class="metric-label">Non-zero px</div><div class="metric-value">{non_zero}</div></div>', unsafe_allow_html=True)
            with m2:
                st.markdown(f'<div class="metric-box"><div class="metric-label">Max value</div><div class="metric-value">{max_val}</div></div>', unsafe_allow_html=True)
            with m3:
                st.markdown(f'<div class="metric-box"><div class="metric-label">Mean (ink)</div><div class="metric-value">{mean_val}</div></div>', unsafe_allow_html=True)
            
            # Download buttons
            st.markdown('<p class="section-header">Export</p>', unsafe_allow_html=True)
            
            dl1, dl2 = st.columns(2)
            
            with dl1:
                buf = io.BytesIO()
                img_resized.save(buf, format="PNG")
                st.download_button("⬇ Download PNG", buf.getvalue(), "mnist_digit.png", "image/png")
            
            with dl2:
                csv_data = '\n'.join([','.join(map(str, row)) for row in img_array_28])
                st.download_button("⬇ Download CSV", csv_data, "mnist_digit.csv", "text/csv")
            
            # Pixel grid
            st.markdown('<p class="section-header">Raw Pixel Values (28×28)</p>', unsafe_allow_html=True)
            df_pixels = np.array(img_array_28)
            st.dataframe(df_pixels, height=200)
            
        else:
            st.markdown("""
            <div style="height:280px; display:flex; align-items:center; justify-content:center; 
                        border: 1px dashed #222; border-radius:8px; color:#333; 
                        font-family:'Space Mono',monospace; font-size:0.8rem; text-align:center;">
                Draw a digit on the left<br>to see MNIST output
            </div>
            """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div style="height:280px; display:flex; align-items:center; justify-content:center; 
                    border: 1px dashed #222; border-radius:8px; color:#333; 
                    font-family:'Space Mono',monospace; font-size:0.8rem; text-align:center;">
            Draw a digit on the left<br>to see MNIST output
        </div>
        """, unsafe_allow_html=True)

st.markdown("---")
st.markdown("""
<p style="font-family:'Space Mono',monospace; color:#333; font-size:0.7rem; text-align:center;">
MNIST Standard · 28×28px · Grayscale · Anti-aliased · Pixel values 0–255
</p>
""", unsafe_allow_html=True)
