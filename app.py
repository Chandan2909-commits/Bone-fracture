%%writefile app.py
import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image

# ✅ Set Streamlit Page Configuration
st.set_page_config(page_title="Bone Fracture Detector", page_icon="🦴", layout="wide")

# ✅ Load TensorFlow Lite Model
@st.cache_resource
def load_tflite_model():
    try:
        interpreter = tf.lite.Interpreter(model_path="bone_fracture_detector.tflite")
        interpreter.allocate_tensors()
        return interpreter
    except Exception as e:
        st.error(f"⚠️ Error loading model: {e}")
        return None

interpreter = load_tflite_model()

# ✅ Function to preprocess image
def preprocess_image(image):
    image = image.convert("RGB")  # Ensure RGB format
    image = image.resize((150, 150))  # Resize to match model training
    image = np.array(image, dtype=np.float32) / 255.0  # Normalize (0-1 scale)
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    return image

# ✅ Function to make prediction using TensorFlow Lite
def predict_tflite(interpreter, image):
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    interpreter.set_tensor(input_details[0]['index'], image)
    interpreter.invoke()
    
    prediction = interpreter.get_tensor(output_details[0]['index'])
    return prediction[0][0]  # Extract confidence score

# ✅ Streamlit UI
st.markdown("""
    <h1 style="text-align:center; color:#0066cc;">🦴 Bone Fracture Detection AI</h1>
    <p style="text-align:center;">Upload an X-ray image and let AI detect if a bone fracture is present.</p>
""", unsafe_allow_html=True)

uploaded_file = st.file_uploader("Upload Bone X-ray", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    try:
        image = Image.open(uploaded_file)

        col1, col2 = st.columns([1, 2])
        with col1:
            st.image(image, caption="Uploaded X-ray", use_column_width=True)

        with col2:
            st.info("Analyzing the X-ray...")

            if interpreter is not None:
                # ✅ Process the image
                processed_image = preprocess_image(image)

                # ✅ Make prediction
                confidence = predict_tflite(interpreter, processed_image)

                # ✅ Define class labels
                predicted_class = "FRACTURE" if confidence > 0.5 else "NORMAL"

                # ✅ Display results
                if confidence > 0.5:
                    st.error(f"⚠️ Bone Fracture Detected (Confidence: {confidence:.2%})")
                else:
                    st.success(f"✅ No Fracture Detected (Confidence: {1 - confidence:.2%})")
            else:
                st.error("⚠️ Model is not loaded. Please check your model file.")

    except Exception as e:
        st.error(f"⚠️ Error processing image: {e}")

# ✅ Custom Button Styling
st.markdown(
    "<style>div.stButton > button {background-color: #0066cc; color: white; padding: 10px 20px; border-radius: 10px;}</style>",
    unsafe_allow_html=True
)
