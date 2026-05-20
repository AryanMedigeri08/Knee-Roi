import cv2
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image


# ROI Extraction 
def extract_roi(pil_img):
    img_np = np.array(pil_img.convert("L"))  # grayscale

    # Step 1: Intensity normalization
    mean, std = img_np.mean(), img_np.std()
    normalized = np.clip((img_np - mean) / (std + 1e-6), 0, 1)
    normalized_uint8 = (normalized * 255).astype(np.uint8)

    # Step 2: Otsu's thresholding
    _, binary = cv2.threshold(
        normalized_uint8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    # Step 3: Morphological closing
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    refined = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

    # Step 4: Connected component analysis — pick largest
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(refined)
    if num_labels <= 1:
        return pil_img, normalized_uint8, binary, refined  # fallback

    largest = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
    x, y, w, h = (
        stats[largest, cv2.CC_STAT_LEFT],
        stats[largest, cv2.CC_STAT_TOP],
        stats[largest, cv2.CC_STAT_WIDTH],
        stats[largest, cv2.CC_STAT_HEIGHT],
    )

    # Step 5: Crop bounding box
    roi = pil_img.crop((x, y, x + w, y + h))
    return roi, normalized_uint8, binary, refined


# Grad-CAM
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


# Page config
st.set_page_config(
    page_title="Knee OA Severity Analysis | MIT-AOE",
    page_icon="🦴",
)

class_names = ["Healthy", "Doubtful", "Minimal", "Moderate", "Severe"]
target_size = (224, 224)

model = tf.keras.models.load_model("./src/models/model_Xception_ft.hdf5")

grad_model = tf.keras.models.clone_model(model)
grad_model.set_weights(model.get_weights())
grad_model.layers[-1].activation = None
grad_model = tf.keras.models.Model(
    inputs=[grad_model.inputs],
    outputs=[
        grad_model.get_layer("global_average_pooling2d_1").input,
        grad_model.output,
    ],
)

# Sidebar
with st.sidebar:
    st.markdown("## 🦴 Knee OA Analyzer")
    st.markdown(
        """
        **A Lightweight, Transparent and Risk-Aware ROI Extraction 
        Framework for Responsible Knee X-Ray Analysis**
        """
    )
    st.divider()
    st.markdown(
        """
        **Authors**  
        Arnav Shende · Aryan Medigeri  
        Sakshi Sharan · Laxmi Kale  
        Suyoga Bansode  

        **Institution**  
        MIT Academy of Engineering, Pune  
        Dept. of CSE (Data Science)
        """
    )
    st.divider()
    st.subheader("⬆️ Upload X-Ray")
    uploaded_file = st.file_uploader("Choose a knee X-ray image")

# Header
st.title("🦴 Knee Osteoarthritis Severity Analysis")
st.caption(
    "MIT Academy of Engineering, Pune · CSE (Data Science) · "
    "Deterministic ROI Extraction + Deep Learning Classification"
)
st.divider()

y_pred = None

if uploaded_file is not None:
    pil_img = Image.open(uploaded_file)

    # Step 1: ROI Extraction 
    st.subheader("Step 1 · ROI Extraction Pipeline")
    st.caption(
        "Deterministic classical image processing — no training data required."
    )

    roi_img, norm_img, binary_img, refined_img = extract_roi(pil_img)

    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        st.image(pil_img, caption="Input X-Ray", use_container_width=True)
    with c2:
        st.image(norm_img, caption="Normalized", use_container_width=True)
    with c3:
        st.image(binary_img, caption="Otsu Threshold", use_container_width=True)
    with c4:
        st.image(refined_img, caption="Morphological Closing", use_container_width=True)
    with c5:
        st.image(roi_img, caption="Extracted ROI ✅", use_container_width=True)

    st.divider()

    # Step 2: Classification
    st.subheader("Step 2 · OA Severity Classification")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Input to CNN (ROI)**")
        st.image(roi_img, use_container_width=True)

        img_tensor = tf.keras.preprocessing.image.img_to_array(
            roi_img.convert("RGB").resize(target_size)
        )
        img_aux = img_tensor.copy()

        if st.button("🔍 Predict OA Severity"):
            img_array = np.expand_dims(img_aux, axis=0)
            img_array = np.float32(img_array)
            img_array = tf.keras.applications.xception.preprocess_input(img_array)

            with st.spinner("Running inference..."):
                y_pred = model.predict(img_array)

            y_pred = 100 * y_pred[0]
            probability = np.amax(y_pred)
            number = np.where(y_pred == np.amax(y_pred))
            grade = str(class_names[np.amax(number)])

            st.success(f"**Predicted Grade: {grade}**")
            st.metric(
                label="Severity Grade",
                value=f"{grade} — {probability:.2f}%",
            )

            # KL grade info
            kl_info = {
                "Healthy": "Grade 0 — No signs of osteoarthritis.",
                "Doubtful": "Grade 1 — Doubtful joint space narrowing.",
                "Minimal": "Grade 2 — Minimal osteophytes.",
                "Moderate": "Grade 3 — Moderate joint space narrowing.",
                "Severe": "Grade 4 — Severe joint space loss.",
            }
            st.info(kl_info[grade])

    if y_pred is not None:
        with col2:
            st.markdown("**Grad-CAM Explainability**")
            heatmap = make_gradcam_heatmap(grad_model, img_array)
            image = save_and_display_gradcam(img_tensor, heatmap)
            st.image(image, use_container_width=True)

            st.markdown("**Class Probability Distribution**")
            fig, ax = plt.subplots(figsize=(5, 2))
            colors = ["#2ecc71" if c == grade else "#3498db" for c in class_names]
            ax.barh(class_names, y_pred, height=0.55, align="center", color=colors)
            for i, (c, p) in enumerate(zip(class_names, y_pred)):
                ax.text(p + 2, i - 0.2, f"{p:.2f}%")
            ax.grid(axis="x")
            ax.set_xlim([0, 120])
            ax.set_xticks(range(0, 101, 20))
            fig.tight_layout()
            st.pyplot(fig)

    st.divider()
    st.caption(
        "Framework: Deterministic ROI Extraction + Xception CNN · "
        "Dice: 0.94 · IoU: 0.92 · Avg. processing: 4.49ms/image"
    )