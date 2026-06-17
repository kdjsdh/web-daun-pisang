import streamlit as st 
import cv2
import numpy as np
import pandas as pd
import mahotas as mt
import joblib
from sklearn.preprocessing import StandardScaler, scale
from sklearn import svm

# Load model dan scaler
@st.cache_resource
def load_model():

    model = joblib.load("svm_model.pkl")
    scaler = joblib.load("scaler.pkl")

    return model, scaler

model, scaler = load_model()

# =========================================
# EKSTRAKSI FITUR
# =========================================

def extract_features_from_image(img):

    # Resize
    img = cv2.resize(img, (224, 224))

    # BGR ke RGB
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Grayscale
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)

    # Blur ringan
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    # =====================================
    # RGB FEATURES
    # =====================================

    red_channel = rgb[:, :, 0]
    green_channel = rgb[:, :, 1]
    blue_channel = rgb[:, :, 2]

    red_mean = np.mean(red_channel)
    green_mean = np.mean(green_channel)
    blue_mean = np.mean(blue_channel)

    red_std = np.std(red_channel)
    green_std = np.std(green_channel)
    blue_std = np.std(blue_channel)

    # =====================================
    # HARALICK FEATURES
    # =====================================

    textures = mt.features.haralick(blur)

    ht_mean = textures.mean(axis=0)

    contrast = ht_mean[1]
    correlation = ht_mean[2]
    inverse_diff_moments = ht_mean[4]
    entropy = ht_mean[8]

    # =====================================
    # FEATURE VECTOR
    # =====================================
    vector = [red_mean,green_mean,blue_mean,red_std,green_std,blue_std,contrast,correlation,inverse_diff_moments,entropy
    ]
    return np.array(vector).reshape(1, -1)

# =========================================
# PREDIKSI
# =========================================

def predict_image(img):

    # Ekstraksi fitur
    features = extract_features_from_image(img)

    # Scaling
    features_scaled = scaler.transform(features)

    # Prediksi
    prediction = model.predict(features_scaled)[0]

    # Confidence
    probabilities = model.predict_proba(features_scaled)

    confidence = np.max(probabilities) * 100

    # Label
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


st.set_page_config(
    page_title="Klasifikasi Penyakit Tanaman pisang",
    page_icon=":leaves:",
    layout="wide",
    initial_sidebar_state="expanded",
)
st.title("Klasifikasi Penyakit Tanaman Pisang")


#uploaded_file = st.file_uploader("Pilih file gambar", type=["jpg", "jpeg", "png"])

#st.sidebar.image("profil.jpg")

# Add instructions on how to use the app to the sidebar
#st.sidebar.header("cara menggunakan Website")

uploaded_file = st.file_uploader(
    "Upload Gambar",
    type=['jpg', 'png', 'jpeg']
)

if uploaded_file is not None:

    # Convert file menjadi array
    file_bytes = np.asarray(bytearray(uploaded_file.read()),dtype=np.uint8
    )

    # Decode image
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    # Tampilkan gambar
    st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB),
        caption="Gambar Input",
        width=300
    )

    # Prediksi
    with st.spinner("Melakukan klasifikasi..."):

        label, confidence = predict_image(img)

    # Hasil
    st.success(f"Hasil Prediksi : {label}")

    st.info(f"Tingkat Keyakinan : {confidence:.2f}%")
