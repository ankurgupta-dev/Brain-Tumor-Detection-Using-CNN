import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
import os

st.set_page_config(page_title="Brain Tumour Detection", page_icon="🧠")

st.title("🧠 Brain Tumour Detection Using CNN")
st.write("Upload a Brain MRI image to predict whether a tumour is present.")

# Load model
model_path = "models/cnn-parameters-improvement-05-0.84.keras"

if not os.path.exists(model_path):
    st.error(f"Model not found: {model_path}")
    st.stop()

model = tf.keras.models.load_model(model_path)

uploaded_file = st.file_uploader(
    "Choose an MRI image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    st.image(img_rgb, caption="Uploaded Image", use_container_width=True)

    img_resized = cv2.resize(img_rgb, (240, 240))
    img_resized = img_resized / 255.0
    img_resized = np.expand_dims(img_resized, axis=0)

    prediction = model.predict(img_resized)[0][0]

    st.subheader("Prediction")

    if prediction >= 0.5:
        st.error(f"🟥 Tumour Detected ({prediction:.2%} confidence)")
    else:
        st.success(f"🟩 No Tumour Detected ({(1-prediction):.2%} confidence)")