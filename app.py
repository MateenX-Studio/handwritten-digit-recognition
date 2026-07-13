import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
from streamlit_drawable_canvas import st_canvas
import cv2

st.set_page_config(page_title="Handwritten Digit Recognition", page_icon="🔢")

@st.cache_resource
def load_model():
    return tf.keras.models.load_model('mnist_model.keras')

model = load_model()

st.title("🔢 Handwritten Digit Recognition System")
st.markdown("**Deep Learning Project | CNN Model | MNIST Dataset | Accuracy: 98.93%**")
st.markdown("---")

def predict_single(gray_img):
    """Single 28x28 gray image se predict karo"""
    resized = cv2.resize(gray_img, (28, 28))
    arr = resized.astype('float32') / 255.0
    arr = arr.reshape(1, 28, 28, 1)
    pred = model.predict(arr, verbose=0)
    return int(np.argmax(pred)), float(np.max(pred)) * 100

def detect_and_predict(pil_image):
    """Image se multiple digits detect karo"""
    gray = np.array(pil_image.convert('L'))

    # Resize if large
    h, w = gray.shape
    if max(h, w) > 800:
        scale = 800 / max(h, w)
        gray = cv2.resize(gray, (int(w*scale), int(h*scale)))

    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # OTSU threshold
    _, binary = cv2.threshold(
        blurred, 0, 255,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )

    # Light background fix
    if np.mean(gray) > 127:
        pass  # OTSU ne already invert kar diya
    else:
        binary = cv2.bitwise_not(binary)

    # Clean up
    kernel = np.ones((2,2), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(
        binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    if not contours:
        return None, "No digit found!"

    ih, iw = binary.shape
    valid = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        area = cv2.contourArea(cnt)
        if area < ih * iw * 0.001: continue
        if w < 8 or h < 8: continue
        if w > iw * 0.95: continue
        if w > h * 6: continue
        valid.append((x, y, w, h))

    if not valid:
        return None, "Could not detect digit clearly!"

    valid = sorted(valid, key=lambda r: r[0])

    results = []
    for (x, y, w, h) in valid:
        pad = 10
        x1, y1 = max(0, x-pad), max(0, y-pad)
        x2, y2 = min(iw, x+w+pad), min(ih, y+h+pad)
        crop = binary[y1:y2, x1:x2]

        dh, dw = crop.shape
        size = max(dh, dw) + 10
        square = np.zeros((size, size), dtype=np.uint8)
        yo, xo = (size-dh)//2, (size-dw)//2
        square[yo:yo+dh, xo:xo+dw] = crop

        digit, conf = predict_single(square)
        results.append({'digit': digit, 'confidence': conf})

    return results, None

def show_results(results):
    number = ''.join([str(r['digit']) for r in results])
    st.success(f"### ✅ Detected Number: **{number}**")
    if len(results) > 1:
        cols = st.columns(len(results))
        for i, r in enumerate(results):
            with cols[i]:
                st.metric(f"Digit {i+1}", r['digit'], f"{r['confidence']:.1f}%")
    else:
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Predicted Digit", results[0]['digit'])
        with col2:
            st.metric("Confidence", f"{results[0]['confidence']:.2f}%")

# ===== TABS =====
tab1, tab2, tab3 = st.tabs(["✏️ Draw", "📷 Camera", "📁 Upload Image"])

# ===== TAB 1: DRAW =====
with tab1:
    st.subheader("✏️ Draw Digit on Canvas")
    st.info("💡 Tip: Draw large and bold digits. Leave space between multiple digits.")

    canvas = st_canvas(
        stroke_width=20,
        stroke_color="white",
        background_color="black",
        height=300,
        width=500,
        drawing_mode="freedraw",
        key="canvas1"
    )

    if st.button("🔍 Predict Digit", key="draw_btn"):
        if canvas.image_data is not None:
            img_arr = canvas.image_data[:, :, :3].astype(np.uint8)
            pil_img = Image.fromarray(img_arr)
            results, error = detect_and_predict(pil_img)
            if error:
                st.warning(f"⚠️ {error}")
            else:
                show_results(results)

# ===== TAB 2: CAMERA =====
with tab2:
    st.subheader("📷 Capture Digit via Camera")
    st.info("💡 Tip: Write a large digit on white paper and hold it in front of the camera.")

    cam_img = st.camera_input("Open Camera")
    if cam_img:
        pil_img = Image.open(cam_img)
        st.image(pil_img, caption="Captured Image", width=300)
        if st.button("🔍 Predict Digit", key="cam_btn"):
            results, error = detect_and_predict(pil_img)
            if error:
                st.warning(f"⚠️ {error}")
            else:
                show_results(results)

# ===== TAB 3: UPLOAD =====
with tab3:
    st.subheader("📁 Upload Image for Prediction")
    st.info("💡 Tip: Best results with black background and white digits.")

    uploaded = st.file_uploader("Choose an image file", type=['png','jpg','jpeg'])
    if uploaded:
        pil_img = Image.open(uploaded)
        st.image(pil_img, caption="Uploaded Image", width=300)
        if st.button("🔍 Predict Digit", key="up_btn"):
            results, error = detect_and_predict(pil_img)
            if error:
                st.warning(f"⚠️ {error}")
            else:
                show_results(results)

st.markdown("---")
st.caption("🤖 Built with CNN | TensorFlow & Keras | MNIST Dataset | Streamlit Web App")