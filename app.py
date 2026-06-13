import streamlit as st
import numpy as np
import pickle
from tensorflow.keras.models import load_model

# Page Config
st.set_page_config(
    page_title="Breast Cancer Prediction",
    page_icon="🩺",
    layout="wide"
)

# Load Model
@st.cache_resource
def load_nn_model():
    return load_model("best_model.keras")

# Load Scaler
@st.cache_resource
def load_scaler():
    with open("scaler.pkl", "rb") as file:
        return pickle.load(file)

model = load_nn_model()
scaler = load_scaler()

st.title("🩺 Breast Cancer Classification using Neural Network")
st.markdown(
    "Enter the tumor characteristics below and click **Predict**."
)

# Feature Names
features = [
    "mean radius",
    "mean texture",
    "mean perimeter",
    "mean area",
    "mean smoothness",
    "mean compactness",
    "mean concavity",
    "mean concave points",
    "mean symmetry",
    "mean fractal dimension",
    "radius error",
    "texture error",
    "perimeter error",
    "area error",
    "smoothness error",
    "compactness error",
    "concavity error",
    "concave points error",
    "symmetry error",
    "fractal dimension error",
    "worst radius",
    "worst texture",
    "worst perimeter",
    "worst area",
    "worst smoothness",
    "worst compactness",
    "worst concavity",
    "worst concave points",
    "worst symmetry",
    "worst fractal dimension"
]

st.subheader("Input Features")

cols = st.columns(3)

input_data = []

for i, feature in enumerate(features):
    with cols[i % 3]:
        value = st.number_input(
            feature,
            min_value=0.0,
            value=0.0,
            format="%.5f"
        )
        input_data.append(value)

if st.button("Predict Cancer Type"):

    input_array = np.array(input_data).reshape(1, -1)

    scaled_input = scaler.transform(input_array)

    prediction = model.predict(scaled_input, verbose=0)

    probability = float(prediction[0][0])

    st.subheader("Prediction Result")

    if probability >= 0.5:
        st.error("🔴 Malignant (Cancer Detected)")
    else:
        st.success("🟢 Benign (Non-Cancerous)")

    st.write(f"Prediction Score: **{probability:.4f}**")

    st.progress(min(probability, 1.0))

st.markdown("---")
st.caption("Built with Streamlit, TensorFlow, and Scikit-Learn")