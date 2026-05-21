import cv2
import matplotlib.cm as cm
import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image
from pathlib import Path
import base64
import textwrap
from datetime import datetime
from urllib.parse import quote
from fpdf import FPDF


# ═══════════════════════════════════════════════════════════════
#  CORE ML FUNCTIONS
# ═══════════════════════════════════════════════════════════════

def extract_roi(pil_img):
    img_np = np.array(pil_img.convert("L"))
    mean, std = img_np.mean(), img_np.std()
    normalized = np.clip((img_np - mean) / (std + 1e-6), 0, 1)
    normalized_uint8 = (normalized * 255).astype(np.uint8)
    _, binary = cv2.threshold(normalized_uint8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    refined = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(refined)
    if num_labels <= 1:
        return pil_img, normalized_uint8, binary, refined
    largest = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
    x, y, w, h = (
        stats[largest, cv2.CC_STAT_LEFT],
        stats[largest, cv2.CC_STAT_TOP],
        stats[largest, cv2.CC_STAT_WIDTH],
        stats[largest, cv2.CC_STAT_HEIGHT],
    )
    roi = pil_img.crop((x, y, x + w, y + h))
    return roi, normalized_uint8, binary, refined


def make_gradcam_heatmap(grad_model, img_array, pred_index=None):
    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(preds[0])
        class_channel = preds[:, pred_index]
    grads = tape.gradient(class_channel, last_conv_layer_output)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()


def save_and_display_gradcam(img, heatmap, alpha=0.4):
    heatmap = np.uint8(255 * heatmap)
    jet = cm.get_cmap("jet")
    jet_colors = jet(np.arange(256))[:, :3]
    jet_heatmap = jet_colors[heatmap]
    jet_heatmap = tf.keras.preprocessing.image.array_to_img(jet_heatmap)
    jet_heatmap = jet_heatmap.resize((img.shape[1], img.shape[0]))
    jet_heatmap = tf.keras.preprocessing.image.img_to_array(jet_heatmap)
    superimposed_img = jet_heatmap * alpha + img
    return tf.keras.preprocessing.image.array_to_img(superimposed_img)


# ═══════════════════════════════════════════════════════════════
#  HTML HELPER
# ═══════════════════════════════════════════════════════════════

def html(text):
    st.markdown(textwrap.dedent(text).strip(), unsafe_allow_html=True)


def get_image_base64(pil_img=None, path=None):
    if path:
        with open(path, "rb") as f:
            return base64.b64encode(f.read()).decode()
    if pil_img:
        import io
        buf = io.BytesIO()
        pil_img.save(buf, format="PNG")
        return base64.b64encode(buf.getvalue()).decode()
    return ""


# ═══════════════════════════════════════════════════════════════
#  UI RENDERING FUNCTIONS
# ═══════════════════════════════════════════════════════════════

def get_severity_color(grade):
    return {"Healthy": "#22C55E", "Doubtful": "#F59E0B", "Minimal": "#F97316", "Moderate": "#F5811F", "Severe": "#EF4444"}.get(grade, "#3B5BDB")

def get_kl_info(grade):
    return {"Healthy": (0, "No signs of osteoarthritis"), "Doubtful": (1, "Doubtful joint space narrowing"), "Minimal": (2, "Minimal osteophytes present"), "Moderate": (3, "Moderate joint space narrowing"), "Severe": (4, "Severe joint space loss")}.get(grade, (0, ""))

def get_grade_description(grade):
    return {"Healthy": "No signs of osteoarthritis detected. The joint space appears normal with no visible osteophytes.", "Doubtful": "Doubtful signs of osteoarthritis. There may be possible joint space narrowing but findings are inconclusive.", "Minimal": "Minimal signs of osteoarthritis detected. Small osteophytes are present with possible joint space narrowing.", "Moderate": "Moderate signs of osteoarthritis detected. There is noticeable joint space narrowing and presence of osteophytes.", "Severe": "Severe osteoarthritis detected. Significant joint space loss with large osteophytes and possible bone deformity."}.get(grade, "")

def get_findings_for_grade(grade):
    severity_map = {
        "Healthy":  {"Joint Space Narrowing": "Normal", "Osteophyte Formation": "Normal", "Sclerosis": "Normal", "Alignment": "Normal"},
        "Doubtful": {"Joint Space Narrowing": "Mild", "Osteophyte Formation": "Normal", "Sclerosis": "Normal", "Alignment": "Normal"},
        "Minimal":  {"Joint Space Narrowing": "Mild", "Osteophyte Formation": "Mild", "Sclerosis": "Normal", "Alignment": "Normal"},
        "Moderate": {"Joint Space Narrowing": "Moderate", "Osteophyte Formation": "Moderate", "Sclerosis": "Mild", "Alignment": "Normal"},
        "Severe":   {"Joint Space Narrowing": "Severe", "Osteophyte Formation": "Severe", "Sclerosis": "Moderate", "Alignment": "Mild"},
    }
    return severity_map.get(grade, severity_map["Healthy"])


def create_report_text(grade, probability):
    kl_num_val, _ = get_kl_info(grade)
    lines = [
        "Knee X-ray Analysis Report",
        f"Generated: {datetime.now().strftime('%B %d, %Y %I:%M %p')}",
        "",
        f"Severity Grade: {grade}",
        f"Overall Severity: {probability:.0f}%",
        f"Kellgren-Lawrence Grade: {kl_num_val}",
        "",
        "Summary:",
        get_grade_description(grade),
        "",
        "Key Findings:",
    ]
    for name, status in get_findings_for_grade(grade).items():
        lines.append(f"- {name}: {status}")
    lines.extend([
        "",
        "Note: This AI-assisted report is for screening support and should be reviewed by a qualified clinician.",
    ])
    return "\n".join(lines)


def generate_pdf_bytes(report_text):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Helvetica", size=12)
    for line in report_text.split('\n'):
        pdf.cell(0, 10, txt=line, ln=True)
    return bytes(pdf.output())


def render_results_panel(grade, probability):
    color = get_severity_color(grade)
    desc = get_grade_description(grade)
    kl_num_val, _ = get_kl_info(grade)
    circ = 282
    offset = circ - (probability / 100) * circ
    pos = min(max(probability, 2), 98)
    
    findings = get_findings_for_grade(grade)
    icons = {"Joint Space Narrowing": "&#129468;", "Osteophyte Formation": "&#128300;", "Sclerosis": "&#129659;", "Alignment": "&#128208;"}
    status_cls = {"Normal": "fs-normal", "Mild": "fs-mild", "Moderate": "fs-moderate", "Severe": "fs-severe"}
    
    findings_html = ""
    for name, status in findings.items():
        findings_html += f'<div class="find-row"><div class="find-ico">{icons[name]}</div><div class="find-name">{name}</div><div class="find-status {status_cls[status]}">{status}</div></div>'

    html(f"""
<div class="results-card anim">
<div class="sev-grade-row">
<div class="sev-grade-left">
<div class="sev-label">Severity Grade</div>
<div class="sev-value" style="color:{color};">{grade}</div>
<div class="kl-row"><span class="kl-label">Kellgren-Lawrence Grade</span> <br><span class="kl-num" style="color:{color};">{kl_num_val}</span></div>
<div class="sev-desc">{desc}</div>
</div>
<div class="mini-ring-wrap">
<svg width="95" height="95" viewBox="0 0 100 100">
<circle cx="50" cy="50" r="45" fill="none" stroke="#F3F4F6" stroke-width="8"/>
<circle cx="50" cy="50" r="45" fill="none" stroke="{color}" stroke-width="8" stroke-linecap="round" stroke-dasharray="{circ}" stroke-dashoffset="{offset}" transform="rotate(-90 50 50)" style="animation: ringAnim 1.5s ease-out;" />
</svg>
<div class="ring-center">
<span class="ring-pct">{probability:.0f}%</span>
<span class="ring-sub">Overall Severity</span>
</div>
</div>
</div>

<div class="scale-section">
<div class="scale-title">Severity Scale &#8505;&#65039;</div>
<div class="scale-bar-wrap">
<div class="scale-bar"></div>
<div class="scale-dot" style="left:{pos}%; border-color:{color};"></div>
</div>
<div class="scale-labels">
<span>Normal<br>(0-15%)</span>
<span>Mild<br>(16-35%)</span>
<span>Moderate<br>(36-65%)</span>
<span>Severe<br>(66-100%)</span>
</div>
</div>

<div class="findings-box">
<div class="findings-hdr">Key Findings</div>
{findings_html}
</div>

</div>
""")


def render_ai_checks():
    checks = [("&#129468;", "Joint Space Narrowing"), ("&#128300;", "Osteophyte Formation"), ("&#129659;", "Sclerosis"), ("&#128208;", "Alignment &amp; Structure")]
    items = ""
    for icon, name in checks:
        items += f'<div class="check-row"><div class="check-ico">{icon}</div><div class="check-label">{name}</div><div class="check-tick">&#10003;</div></div>'
    html(f'<div class="results-card anim" style="margin-top:1.5rem;"><div class="checks-title">Our <strong>AI checks for</strong></div>{items}</div>')


def render_tip():
    html('<div class="tip-box anim" style="animation-delay:0.1s;"><div class="tip-icon">&#128161;</div><div><div class="tip-title">Tip</div><div class="tip-text">Please ensure a clear front view X-ray for best results.</div></div></div>')


def render_xray_progress(image_b64, percent):
    html(f"""
<div class="xray-progress-wrap anim" style="animation-delay: 0.1s;">
<img src="data:image/png;base64,{image_b64}" />
<div class="analysis-ring" style="--progress:{percent:.0f};">
<svg viewBox="0 0 180 180">
<circle class="analysis-track" cx="90" cy="90" r="72"></circle>
<circle class="analysis-meter" cx="90" cy="90" r="72"></circle>
</svg>
<div class="analysis-center">
<strong>{percent:.0f}%</strong>
<span>Analyzing...</span>
</div>
</div>
</div>
    """)


# ═══════════════════════════════════════════════════════════════
#  PAGE CONFIG & SETUP
# ═══════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="Knee X-ray Analysis | AI-Powered",
    page_icon="🦴",
    layout="wide",
    initial_sidebar_state="expanded",
)

css_path = Path(__file__).parent / "custom_theme.css"
with open(css_path) as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

class_names = ["Healthy", "Doubtful", "Minimal", "Moderate", "Severe"]
target_size = (224, 224)

@st.cache_resource
def load_models():
    model = tf.keras.models.load_model("./src/models/model_Xception_ft.hdf5")
    grad_model = tf.keras.models.clone_model(model)
    grad_model.set_weights(model.get_weights())
    grad_model.layers[-1].activation = None
    grad_model = tf.keras.models.Model(
        inputs=[grad_model.inputs],
        outputs=[grad_model.get_layer("global_average_pooling2d_1").input, grad_model.output],
    )
    return model, grad_model

model, grad_model = load_models()

for key in ["y_pred", "grade", "img_tensor", "img_array", "analyzed", "show_progress", "report_text", "uploaded_name", "active_tab", "roi_img", "norm_img", "binary_img", "refined_img", "pil_img_original"]:
    if key not in st.session_state:
        if key == "active_tab":
            st.session_state[key] = "Home"
        else:
            st.session_state[key] = None


# ═══════════════════════════════════════════════════════════════
#  SIDEBAR
# ═══════════════════════════════════════════════════════════════

with st.sidebar:
    home_icon = get_image_base64(path=Path(__file__).parent / "img" / "home.png")
    history_icon = get_image_base64(path=Path(__file__).parent / "img" / "file.png")
    insights_icon = get_image_base64(path=Path(__file__).parent / "img" / "report.png")

    active_idx = ["Home", "Insights", "History"].index(st.session_state.active_tab) + 1
    html(f"""
    <style>
    div[data-testid="stRadio"] label:nth-child(1) div::before {{ background-image: url(data:image/png;base64,{home_icon}); }}
    div[data-testid="stRadio"] label:nth-child(2) div::before {{ background-image: url(data:image/png;base64,{insights_icon}); }}
    div[data-testid="stRadio"] label:nth-child(3) div::before {{ background-image: url(data:image/png;base64,{history_icon}); }}
    div[data-testid="stRadio"] label:nth-child({active_idx}) {{
        background: var(--blue-light) !important;
        color: var(--blue) !important;
    }}
    </style>
    """)

    html("""
<div class="sidebar-logo">
<span><svg viewBox="0 0 48 48" fill="none"><path d="M23 3l4.5 12.5L40 20l-12.5 4.5L23 37l-4.5-12.5L6 20l12.5-4.5L23 3Z" fill="currentColor"/><path d="M39 5l1.4 3.6L44 10l-3.6 1.4L39 15l-1.4-3.6L34 10l3.6-1.4L39 5Z" fill="currentColor"/></svg></span>
</div>
<div class="sidebar-hamburger"><svg viewBox="0 0 24 24" fill="none"><path d="M4 6h16M4 12h16M4 18h16" stroke="currentColor" stroke-width="2.2" stroke-linecap="round"/></svg></div>
    """)

    st.session_state.active_tab = st.radio("Navigation", ["Home", "Insights", "History"], label_visibility="collapsed", index=["Home", "Insights", "History"].index(st.session_state.active_tab))


# ═══════════════════════════════════════════════════════════════
#  MAIN APP
# ═══════════════════════════════════════════════════════════════

# Notification bell
bell_b64 = get_image_base64(path=Path(__file__).parent / "img" / "bell.png")
html(f"""
<div class="notif-bell">
<img src="data:image/png;base64,{bell_b64}" />
<span></span>
</div>
""")

if st.session_state.active_tab == "Home":
    hero_path = Path(__file__).parent / "img" / "hero_knee.png"
    hero_b64 = get_image_base64(path=hero_path)
    
    html(f"""
<div class="hero-banner anim">
<div class="hero-text">
<h1 class="hero-title">Knee X-ray <span style="color:var(--blue);">Analysis</span></h1>
<p class="hero-subtitle">AI-powered insights for better knee health</p>
</div>
<img src="data:image/png;base64,{hero_b64}" class="hero-img" alt="Hero">
</div>
    """)

    html("""
<div class="upload-area anim" style="animation-delay: 0.1s;">
<div class="upload-circle">
<svg viewBox="0 0 24 24" fill="none"><path d="M12 16V4M12 4L8 8M12 4L16 8" stroke="white" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"/><path d="M20 16V18C20 19.1 19.1 20 18 20H6C4.9 20 4 19.1 4 18V16" stroke="white" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"/></svg>
</div>
<div class="upload-title">Upload Knee X-ray</div>
<div class="upload-formats">JPG, PNG or DICOM</div>
    """)
    
    uploaded_file = st.file_uploader("Choose file", type=["jpg", "jpeg", "png"], label_visibility="collapsed")
    
    html('<div class="upload-note">or drag and drop</div></div>')
    
    if uploaded_file is not None:
        if st.session_state.uploaded_name != uploaded_file.name:
            st.session_state.uploaded_name = uploaded_file.name
            st.session_state.analyzed = False
            pil_img = Image.open(uploaded_file)
            st.session_state.pil_img_original = pil_img
            roi_img, norm, binary, refined = extract_roi(pil_img)
            st.session_state.roi_img = roi_img
            st.session_state.norm_img = Image.fromarray(norm)
            st.session_state.binary_img = Image.fromarray(binary)
            st.session_state.refined_img = Image.fromarray(refined)
            
        st.image(st.session_state.pil_img_original, width=300, caption="Uploaded Image")
        
        if st.button("Analyze X-Ray", use_container_width=True):
            with st.spinner("Analyzing your X-ray..."):
                img_tensor = tf.keras.preprocessing.image.img_to_array(st.session_state.roi_img.convert("RGB").resize(target_size))
                st.session_state.img_tensor = img_tensor
                img_array = np.expand_dims(img_tensor.copy(), axis=0)
                img_array = np.float32(img_array)
                img_array = tf.keras.applications.xception.preprocess_input(img_array)
                st.session_state.img_array = img_array
                
                y_pred = model.predict(img_array)
                y_pred = 100 * y_pred[0]
                number = np.where(y_pred == np.amax(y_pred))
                
                st.session_state.y_pred = y_pred
                st.session_state.grade = str(class_names[np.amax(number)])
                st.session_state.report_text = create_report_text(st.session_state.grade, float(np.amax(y_pred)))
                st.session_state.analyzed = True
                st.session_state.active_tab = "Insights"
                st.rerun()

elif st.session_state.active_tab == "Insights":
    if not st.session_state.analyzed:
        st.info("Please upload and analyze an image from the Home tab first.")
    else:
        col1, col2 = st.columns([1.5, 1], gap="large")
        with col1:
            st.markdown("### Step 1 &bull; ROI Extraction Pipeline")
            st.markdown("Deterministic classical image processing – no training data required.")
            
            step1_cols = st.columns(5)
            with step1_cols[0]: st.image(st.session_state.pil_img_original, caption="Input X-Ray", use_container_width=True)
            with step1_cols[1]: st.image(st.session_state.norm_img, caption="Normalized", use_container_width=True)
            with step1_cols[2]: st.image(st.session_state.binary_img, caption="Otsu Threshold", use_container_width=True)
            with step1_cols[3]: st.image(st.session_state.refined_img, caption="Morphological Closing", use_container_width=True)
            with step1_cols[4]: st.image(st.session_state.roi_img, caption="Extracted ROI ✅", use_container_width=True)
            
            st.markdown("<br>### Step 2 &bull; OA Severity Classification", unsafe_allow_html=True)
            st.markdown("Input to CNN (ROI) & Grad-CAM Explainability")
            
            step2_cols = st.columns(2)
            with step2_cols[0]:
                st.image(st.session_state.roi_img, caption="ROI", use_container_width=True)
            with step2_cols[1]:
                heatmap = make_gradcam_heatmap(grad_model, st.session_state.img_array)
                gradcam_img = save_and_display_gradcam(st.session_state.img_tensor, heatmap)
                st.image(gradcam_img, caption="Grad-CAM", use_container_width=True)
                
        with col2:
            st.markdown("### Analysis Result")
            grade = st.session_state.grade
            probability = np.amax(st.session_state.y_pred)
            render_results_panel(grade, probability)
            
            st.markdown("<br>#### Class Probability Distribution", unsafe_allow_html=True)
            chart_html = ""
            for i, name in enumerate(class_names):
                val = st.session_state.y_pred[i]
                color = get_severity_color(name)
                chart_html += f'<div style="display:flex; align-items:center; margin-bottom:10px;"><div style="width:70px;font-size:0.85rem;color:var(--text-body);font-weight:600;">{name}</div><div style="flex:1;background:#f0f0f0;height:14px;border-radius:7px;margin:0 12px;overflow:hidden;"><div style="width:{val}%;background:{color};height:100%;border-radius:7px;"></div></div><div style="width:50px;font-size:0.85rem;color:var(--text-body);font-weight:600;text-align:right;">{val:.2f}%</div></div>'
            st.markdown(chart_html, unsafe_allow_html=True)

            st.markdown('<div class="report-actions">', unsafe_allow_html=True)
            share_url = "mailto:?subject=Knee%20X-ray%20Analysis%20Report&body=" + quote(st.session_state.report_text)
            action_col1, action_col2 = st.columns(2, gap="medium")
            with action_col1:
                st.link_button("Share Report", share_url, use_container_width=True)
            with action_col2:
                pdf_bytes = generate_pdf_bytes(st.session_state.report_text)
                st.download_button("Download Report", data=pdf_bytes, file_name="knee_xray_analysis_report.pdf", mime="application/pdf", use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

elif st.session_state.active_tab == "History":
    hero_path = Path(__file__).parent / "img" / "hero_knee.png"
    hero_thumb_b64 = get_image_base64(path=hero_path)
    html(f"""
<div class="anim" style="max-width: 800px; margin: 0 auto; animation-delay: 0.2s;">
<div class="recent-hdr">
<h3>Recent Analyses</h3>
<a href="#">View All</a>
</div>
<div class="recent-row">
<div class="recent-thumb"><img src="data:image/png;base64,{hero_thumb_b64}"></div>
<div class="recent-meta">
<div class="recent-date">May 20, 2024 &middot; 10:30 AM</div>
<div class="recent-knee">Right Knee</div>
</div>
<div class="recent-badge mild">Mild (24%)</div>
<div class="recent-arrow">&#8250;</div>
</div>
<div class="recent-row">
<div class="recent-thumb"><img src="data:image/png;base64,{hero_thumb_b64}"></div>
<div class="recent-meta">
<div class="recent-date">May 18, 2024 &middot; 02:15 PM</div>
<div class="recent-knee">Left Knee</div>
</div>
<div class="recent-badge moderate" style="background:var(--amber-light);color:var(--orange);">Moderate (52%)</div>
<div class="recent-arrow">&#8250;</div>
</div>
</div>
    """)
