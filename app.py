import streamlit as st
import cv2
import numpy as np
import joblib
from skimage.feature import graycomatrix, graycoprops

# =========================================
# LOAD MODEL & SCALER
# =========================================
@st.cache_resource
def load_model():
    model = joblib.load("svm_model.pkl")
    scaler = joblib.load("scaler.pkl")
    return model, scaler

model, scaler = load_model()

# =========================================
# FEATURE EXTRACTION (FIX FINAL)
# =========================================
def extract_features_from_image(img):

    # Resize
    img = cv2.resize(img, (224, 224))

    # RGB
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Gray
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)

    # Blur
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    # =====================================
    # RGB FEATURES
    # =====================================
    red_mean = np.mean(rgb[:, :, 0])
    green_mean = np.mean(rgb[:, :, 1])
    blue_mean = np.mean(rgb[:, :, 2])

    red_std = np.std(rgb[:, :, 0])
    green_std = np.std(rgb[:, :, 1])
    blue_std = np.std(rgb[:, :, 2])

    # =====================================
    # TEXTURE FEATURES (GLCM REPLACEMENT HARALICK)
    # =====================================
    glcm = graycomatrix(
        blur,
        distances=[1],
        angles=[0],
        levels=256,
        symmetric=True,
        normed=True
    )

    contrast = graycoprops(glcm, 'contrast')[0, 0]
    correlation = graycoprops(glcm, 'correlation')[0, 0]
    energy = graycoprops(glcm, 'energy')[0, 0]
    homogeneity = graycoprops(glcm, 'homogeneity')[0, 0]

    # =====================================
    # FEATURE VECTOR (TOTAL 10 FITUR)
    # =====================================
    features = np.array([
        red_mean, green_mean, blue_mean,
        red_std, green_std, blue_std,
        contrast, correlation, energy, homogeneity
    ])

    return features.reshape(1, -1)

# =========================================
# PREDIKSI
# =========================================
def predict_image(img):

    features = extract_features_from_image(img)

    # Scaling
    features_scaled = scaler.transform(features)

    # Prediksi
    prediction = model.predict(features_scaled)[0]

    # Confidence
    probabilities = model.predict_proba(features_scaled)
    confidence = np.max(probabilities) * 100

    # Label mapping
    if prediction == 1:
        label = "Cordana"
    elif prediction == 2:
        label = "Pestalotiopsis"
    elif prediction == 3:
        label = "Sehat"
    elif prediction == 4:
        label = "Sigatoka"
    else:
        label = "Tidak dikenali"

    return label, confidence

# =========================================
# STREAMLIT UI
# =========================================
st.set_page_config(
    page_title="Klasifikasi Penyakit Tanaman Pisang",
    page_icon="🍃",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("🍃 Klasifikasi Penyakit Tanaman Pisang")

uploaded_file = st.file_uploader(
    "Upload Gambar Daun Pisang",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    # Decode image
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    # Tampilkan gambar
    st.image(
        cv2.cvtColor(img, cv2.COLOR_BGR2RGB),
        caption="Gambar Input",
        width=300
    )

    # Prediksi
    with st.spinner("Melakukan klasifikasi..."):
        label, confidence = predict_image(img)

    # Hasil
    st.success(f"Hasil Prediksi: {label}")
    st.info(f"Tingkat Keyakinan: {confidence:.2f}%")
