import streamlit as st
import tensorflow
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image
import os

# Load model
model = load_model('mobilenetv2_waste_classification.h5')

# Set image size and class names
IMG_SIZE = (224, 224)  
TRAIN_DIR = r"C:\Users\ASUS\Desktop\wastes\train"
class_labels = sorted(os.listdir(TRAIN_DIR))

# Title
st.title("♻ Waste Type Classification")
st.write("Upload an image to predict its waste category.")

# Upload file
uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="Uploaded Image", width=250)

    # Preprocess
    img = img.resize(IMG_SIZE)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0

    # Predict
    predictions = model.predict(img_array)
    predicted_class_index = np.argmax(predictions, axis=1)[0]
    predicted_class = class_labels[predicted_class_index]
    confidence = np.max(predictions) * 100

    st.success(f"**Predicted Waste Type:** {predicted_class.upper()}")
    st.info(f"**Confidence:** {confidence:.2f}%")