import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# Set page config
st.set_page_config(page_title="Lung Disease Detection", layout="centered")

# Load model
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("model.h5")

model = load_model()

# Class labels
class_names = ['Lung_Opacity', 'Normal', 'Viral Pneumonia']

# Title
st.title("Lung Disease Detection from X-ray")
st.write("Upload a Chest X-ray image to predict the disease.")

# File uploader
uploaded_file = st.file_uploader("Choose an X-ray image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded X-ray", use_column_width=True)

    # Preprocess image
    img = image.resize((224, 224))
    img_array = tf.keras.utils.img_to_array(img)
    img_array = tf.expand_dims(img_array, axis=0)

    # Prediction
    predictions = model.predict(img_array)
    score = tf.nn.softmax(predictions[0])
    predicted_class = class_names[np.argmax(score)]
    confidence = 100 * np.max(score)

    # Display result
    st.subheader(f"Prediction: {predicted_class}")
    st.write(f"Confidence: {confidence:.2f}%")
