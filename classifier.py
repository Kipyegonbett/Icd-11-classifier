"""
Medical Notes ICD-11 Classifier - Streamlit App
Now with Mock Model Fallback
"""

import os
import re
import pickle
import numpy as np
import streamlit as st
import tensorflow as tf
import plotly.express as px
from tensorflow.keras.preprocessing.sequence import pad_sequences

# -----------------------------------------------------------------------------
# PAGE CONFIG
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="ICD-11 Medical Notes Classifier",
    page_icon="🏥",
    layout="wide",
)

# -----------------------------------------------------------------------------
# MODEL + TOKENIZER + ENCODER LOADING
# -----------------------------------------------------------------------------
@st.cache_resource
def load_model_and_tools():
    """Load TensorFlow model and preprocessing tools, or use mock fallback."""
    model = None
    tokenizer = None
    label_encoder = None

    model_paths = [
        "icd11_classifier_model.keras",
        "icd11_classifier_model.h5",
        "icd11_classifier_model",  # directory
    ]

    # Try loading real model
    for path in model_paths:
        if os.path.exists(path):
            try:
                model = tf.keras.models.load_model(path)
                st.success(f"✅ Loaded model from `{path}`")
                break
            except Exception as e:
                st.warning(f"⚠️ Could not load model from `{path}`: {e}")

    # Try tokenizer / encoder
    if os.path.exists("icd11_tokenizer.pickle"):
        with open("icd11_tokenizer.pickle", "rb") as f:
            tokenizer = pickle.load(f)
    if os.path.exists("icd11_label_encoder.pickle"):
        with open("icd11_label_encoder.pickle", "rb") as f:
            label_encoder = pickle.load(f)

    # If anything is missing, make mock replacements
    if model is None or tokenizer is None or label_encoder is None:
        st.warning("⚠️ Real model or tools not found — using mock classifier for testing.")

        class MockModel:
            def predict(self, x, verbose=0):
                probs = np.random.dirichlet(np.ones(5), size=1)  # 5 random classes
                return probs

        class MockTokenizer:
            def texts_to_sequences(self, texts):
                return [[1, 2, 3, 4, 5]]

        class MockLabelEncoder:
            classes_ = np.array([
                "Diseases of the nervous system",
                "Neoplasms",
                "Endocrine, nutritional or metabolic diseases",
                "Diseases of the circulatory system",
                "Diseases of the respiratory system"
            ])
            def inverse_transform(self, indices):
                return self.classes_[indices]

        model = MockModel()
        tokenizer = MockTokenizer()
        label_encoder = MockLabelEncoder()

    return model, tokenizer, label_encoder


model, tokenizer, label_encoder = load_model_and_tools()

# -----------------------------------------------------------------------------
# ICD CHAPTER METADATA
# -----------------------------------------------------------------------------
ICD_CHAPTERS = {
    "Diseases of the nervous system": {"code": "08", "description": "Neurological conditions", "color": "#cffafe"},
    "Neoplasms": {"code": "02", "description": "Tumors and abnormal tissue growths", "color": "#f3e8ff"},
    "Endocrine, nutritional or metabolic diseases": {"code": "05", "description": "Hormonal and metabolic disorders", "color": "#fef3c7"},
    "Diseases of the circulatory system": {"code": "11", "description": "Heart and blood vessel diseases", "color": "#ffe4e6"},
    "Diseases of the respiratory system": {"code": "12", "description": "Lung and breathing-related conditions", "color": "#e0f2fe"},
}

# -----------------------------------------------------------------------------
# TEXT PREPROCESSING
# -----------------------------------------------------------------------------
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r"[^a-zA-Z0-9\s\-/]", " ", text)
    return " ".join(text.split())

# -----------------------------------------------------------------------------
# PREDICTION
# -----------------------------------------------------------------------------
def predict_category(text, show_top_n=3):
    processed = preprocess_text(text)
    seq = tokenizer.texts_to_sequences([processed])
    padded = pad_sequences(seq, maxlen=600, padding="post", truncating="post")
    prediction = model.predict(padded, verbose=0)
    predicted_class = np.argmax(prediction, axis=1)[0]
    confidence = float(prediction[0][predicted_class])
    category = label_encoder.inverse_transform([predicted_class])[0]

    # top N predictions
    top_idx = np.argsort(prediction[0])[-show_top_n:][::-1]
    top_preds = [
        {"category": label_encoder.inverse_transform([i])[0], "confidence": float(prediction[0][i])}
        for i in top_idx
    ]
    return {"category": category, "confidence": confidence, "top_predictions": top_preds}

# -----------------------------------------------------------------------------
# UI HEADER
# -----------------------------------------------------------------------------
st.markdown(
    """
<div style='text-align:center;padding:25px;border-radius:15px;
background:linear-gradient(135deg,#6366f1 0%,#8b5cf6 100%);color:white;'>
<h1>🏥 ICD-11 Medical Notes Classifier</h1>
<p>AI-powered classification of clinical notes into ICD-11 chapters</p>
</div>
""",
    unsafe_allow_html=True,
)

# -----------------------------------------------------------------------------
# INPUT SECTION
# -----------------------------------------------------------------------------
st.subheader("📄 Input Medical Notes")
uploaded_file = st.file_uploader("Upload a .txt file", type=["txt"])
text_input = st.text_area("Or paste notes below:", height=200)

if uploaded_file is not None:
    text_input = uploaded_file.read().decode("utf-8")

# -----------------------------------------------------------------------------
# CLASSIFICATION BUTTON
# -----------------------------------------------------------------------------
if st.button("🔍 Classify Notes", use_container_width=True):
    if not text_input.strip():
        st.error("⚠️ Please enter or upload medical notes first.")
    else:
        with st.spinner("Analyzing notes..."):
            result = predict_category(text_input)

        cat = result["category"]
        conf = result["confidence"]
        top_preds = result["top_predictions"]
        info = ICD_CHAPTERS.get(cat, {"code": "??", "description": "Unknown", "color": "#f3f4f6"})

        st.markdown(
            f"""
        <div style="background:{info['color']};padding:20px;border-radius:10px;border-left:6px solid #4f46e5;">
        <h3>📊 Primary Classification</h3>
        <h2>{cat}</h2>
        <p><b>ICD-11 Chapter {info['code']}</b> — {info['description']}</p>
        <p style="color:#4338ca;font-weight:bold;">Confidence: {conf*100:.1f}%</p>
        </div>
        """,
            unsafe_allow_html=True,
        )
        st.progress(conf)

        # Plotly bar chart
        st.markdown("### 📈 Top 3 Prediction Confidences")
        data = {
            "ICD-11 Chapter": [p["category"] for p in top_preds],
            "Confidence (%)": [p["confidence"] * 100 for p in top_preds],
        }
        fig = px.bar(
            data,
            x="Confidence (%)",
            y="ICD-11 Chapter",
            orientation="h",
            color="Confidence (%)",
            color_continuous_scale="Blues",
            range_color=[0, 100],
        )
        fig.update_layout(height=400, margin=dict(l=20, r=20, t=40, b=20))
        st.plotly_chart(fig, use_container_width=True)

        # Alternative predictions
        st.markdown("### 🔄 Alternative Classifications")
        for alt in top_preds[1:]:
            desc = ICD_CHAPTERS.get(alt["category"], {}).get("description", "Unknown")
            st.markdown(
                f"""
            <div style="background:#f9fafb;padding:10px;border-radius:8px;border:1px solid #ddd;margin-bottom:8px;">
            <b>{alt['category']}</b> — {desc}<br>
            <small>Confidence: {alt['confidence']*100:.1f}%</small>
            </div>
            """,
                unsafe_allow_html=True,
            )

# -----------------------------------------------------------------------------
# FOOTER
# -----------------------------------------------------------------------------
st.markdown(
    """
---
⚠️ **Disclaimer:** This AI model (or its mock) is for demonstration and educational use only.  
Always verify clinical interpretations with qualified professionals.
"""
)
