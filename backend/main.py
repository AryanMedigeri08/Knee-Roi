import io
import base64
import cv2
import numpy as np
import tensorflow as tf
from PIL import Image
from pathlib import Path
from datetime import datetime
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import matplotlib.cm as cm
from fpdf import FPDF
from fastapi.responses import Response

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ═══════════════════════════════════════════════════════════════
#  MODEL LOADING
# ═══════════════════════════════════════════════════════════════
class_names = ["Healthy", "Doubtful", "Minimal", "Moderate", "Severe"]
target_size = (224, 224)

# Load Model
model_path = Path(__file__).parent.parent / "src" / "models" / "model_Xception_ft.hdf5"
print(f"Loading model from: {model_path}")

try:
    model = tf.keras.models.load_model(str(model_path))
    grad_model = tf.keras.models.clone_model(model)
    grad_model.set_weights(model.get_weights())
    grad_model.layers[-1].activation = None
    grad_model = tf.keras.models.Model(
        inputs=[grad_model.inputs],
        outputs=[grad_model.get_layer("global_average_pooling2d_1").input, grad_model.output],
    )
except Exception as e:
    print(f"Error loading model: {e}")
    model, grad_model = None, None

# ═══════════════════════════════════════════════════════════════
#  ML FUNCTIONS
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

def make_gradcam_heatmap(img_array, pred_index=None):
    if grad_model is None:
        return np.zeros((10, 10))
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

def get_image_base64(pil_img):
    buf = io.BytesIO()
    pil_img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")

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


# ═══════════════════════════════════════════════════════════════
#  ENDPOINTS
# ═══════════════════════════════════════════════════════════════
@app.get("/")
def read_root():
    return {"status": "Backend is running!"}

@app.post("/analyze")
async def analyze_xray(file: UploadFile = File(...)):
    contents = await file.read()
    pil_img = Image.open(io.BytesIO(contents)).convert("RGB")
    
    # 1. ROI Extraction
    roi_img, norm, binary, refined = extract_roi(pil_img)
    
    # 2. Prediction
    img_tensor = tf.keras.preprocessing.image.img_to_array(roi_img.convert("RGB").resize(target_size))
    img_array = np.expand_dims(img_tensor.copy(), axis=0)
    img_array = np.float32(img_array)
    img_array = tf.keras.applications.xception.preprocess_input(img_array)
    
    y_pred = model.predict(img_array)[0] * 100
    pred_idx = np.argmax(y_pred)
    grade = class_names[pred_idx]
    probability = float(np.amax(y_pred))
    
    # 3. Grad-CAM
    heatmap = make_gradcam_heatmap(img_array, pred_index=pred_idx)
    gradcam_img = save_and_display_gradcam(img_tensor, heatmap)
    
    # 4. Report
    report_text = create_report_text(grade, probability)
    findings = get_findings_for_grade(grade)
    kl_num_val, _ = get_kl_info(grade)

    return {
        "success": True,
        "results": {
            "grade": grade,
            "probability": probability,
            "kl_grade": kl_num_val,
            "description": get_grade_description(grade),
            "findings": findings,
            "report_text": report_text,
            "probabilities": {class_names[i]: float(y_pred[i]) for i in range(len(class_names))}
        },
        "images": {
            "input": get_image_base64(pil_img),
            "norm": get_image_base64(Image.fromarray(norm)),
            "otsu": get_image_base64(Image.fromarray(binary)),
            "morph": get_image_base64(Image.fromarray(refined)),
            "roi": get_image_base64(roi_img),
            "gradcam": get_image_base64(gradcam_img)
        }
    }

class ReportRequest(BaseModel):
    report_text: str

@app.post("/download_pdf")
def download_pdf(req: ReportRequest):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Helvetica", size=12)
    for line in req.report_text.split('\n'):
        pdf.cell(0, 10, txt=line, ln=True)
    
    pdf_bytes = bytes(pdf.output())
    return Response(content=pdf_bytes, media_type="application/pdf", headers={
        "Content-Disposition": 'attachment; filename="knee_xray_analysis_report.pdf"'
    })
