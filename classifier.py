"""
Medical Notes ICD-11 Classifier - Streamlit Interface (with Bar Chart)
Updated version: Includes Plotly bar chart for top predictions
"""

import streamlit as st
import tensorflow as tf
import pickle
import numpy as np
import re
import plotly.express as px
from tensorflow.keras.preprocessing.sequence import pad_sequences

# =============================================================================
# PAGE CONFIGURATION
# =============================================================================
st.set_page_config(
    page_title="ICD-11 Medical Notes Classifier",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =============================================================================
# LOAD MODEL AND PREPROCESSORS
# =============================================================================
@st.cache_resource
def load_model_and_tools():
    model = tf.keras.models.load_model('icd11_classifier_model.keras')
    with open('icd11_tokenizer.pickle', 'rb') as f:
        tokenizer = pickle.load(f)
    with open('icd11_label_encoder.pickle', 'rb') as f:
        label_encoder = pickle.load(f)
    return model, tokenizer, label_encoder

model, tokenizer, label_encoder = load_model_and_tools()
vocab_size = len(tokenizer.word_index) + 1

# =============================================================================
# ICD CHAPTER METADATA
# =============================================================================
ICD_CHAPTERS = {
    'Certain infectious or parasitic diseases': {'code': '01', 'description': 'Diseases caused by infectious agents or parasites', 'color': '#fee2e2'},
    'Neoplasms': {'code': '02', 'description': 'Cancers, tumors, and abnormal tissue growths', 'color': '#f3e8ff'},
    'Diseases of the blood or blood-forming organs': {'code': '03', 'description': 'Disorders affecting blood and blood-forming tissues', 'color': '#fce7f3'},
    'Diseases of the immune system': {'code': '04', 'description': 'Immune system disorders and deficiencies', 'color': '#e0e7ff'},
    'Endocrine, nutritional or metabolic diseases': {'code': '05', 'description': 'Hormonal, nutritional, and metabolic disorders', 'color': '#fef3c7'},
    'Mental, behavioural or neurodevelopmental disorders': {'code': '06', 'description': 'Mental health and behavioral conditions', 'color': '#dbeafe'},
    'Sleep-wake disorders': {'code': '07', 'description': 'Sleep disturbances and disorders', 'color': '#f1f5f9'},
    'Diseases of the nervous system': {'code': '08', 'description': 'Neurological conditions affecting the brain and nerves', 'color': '#cffafe'},
    'Diseases of the visual system': {'code': '09', 'description': 'Eye and vision-related disorders', 'color': '#ccfbf1'},
    'Diseases of the ear or mastoid process': {'code': '10', 'description': 'Ear and hearing-related conditions', 'color': '#d1fae5'},
    'Diseases of the circulatory system': {'code': '11', 'description': 'Heart and blood vessel diseases', 'color': '#ffe4e6'},
    'Diseases of the respiratory system': {'code': '12', 'description': 'Lung and breathing-related conditions', 'color': '#e0f2fe'},
    'Diseases of the digestive system': {'code': '13', 'description': 'Gastrointestinal and digestive disorders', 'color': '#ffedd5'},
    'Diseases of the skin': {'code': '14', 'description': 'Skin and subcutaneous tissue conditions', 'color': '#fef08a'},
    'Diseases of the musculoskeletal system or connective tissue': {'code': '15', 'description': 'Bone, joint, and muscle disorders', 'color': '#d9f99d'},
    'Diseases of the genitourinary system': {'code': '16', 'description': 'Urinary and reproductive system disorders', 'color': '#ddd6fe'},
    'Conditions related to sexual health': {'code': '17', 'description': 'Sexual and reproductive health conditions', 'color': '#f5d0fe'},
    'Pregnancy, childbirth or the puerperium': {'code': '18', 'description': 'Conditions related to pregnancy and childbirth', 'color': '#fbcfe8'},
    'Certain conditions originating in the perinatal period': {'code': '19', 'description': 'Newborn and early infant conditions', 'color': '#bfdbfe'},
    'Developmental anomalies': {'code': '20', 'description': 'Congenital malformations and birth defects', 'color': '#a7f3d0'},
    'Symptoms, signs or clinical findings, not elsewhere classified': {'code': '21', 'description': 'General symptoms and clinical findings', 'color': '#e5e7eb'},
    'Injury, poisoning or certain other consequences of external causes': {'code': '22', 'description': 'Injuries, poisoning, and external causes', 'color': '#fecaca'}
}

# =============================================================================
# PREPROCESSING FUNCTION
# =============================================================================
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z0-9\s\-/]', ' ', text)
    text = ' '.join(text.split())
    return text

# =============================================================================
# PREDICTION FUNCTION
# =============================================================================
def predict_category(text, show_top_n=3):
    processed = preprocess_text(text)
    sequence = tokenizer.texts_to_sequences([processed])
    padded = pad_sequences(sequence, maxlen=600, padding='post', truncating='post')
    prediction = model.predict(padded, verbose=0)
    predicted_class = np.argmax(prediction, axis=1)[0]
    confidence = prediction[0][predicted_class]
    category = label_encoder.inverse_transform([predicted_class])[0]
    top_indices = np.argsort(prediction[0])[-show_top_n:][::-1]
    top_predictions = [{'category': label_encoder.inverse_transform([idx])[0], 'confidence': float(prediction[0][idx])} for idx in top_indices]
    return {'category': category, 'confidence': float(confidence), 'top_predictions': top_predictions}

# =============================================================================
# HEADER
# =============================================================================
st.markdown("""
<div style='text-align:center; padding:30px; border-radius:15px; background: linear-gradient(135deg,#667eea 0%,#764ba2 100%); color:white;'>
<h1>🏥 Medical Notes ICD-11 Classifier</h1>
<p>AI-powered classification of clinical notes into ICD-11 chapters</p>
</div>
""", unsafe_allow_html=True)

# =============================================================================
# SIDEBAR
# =============================================================================
st.sidebar.header("📘 How to Use")
st.sidebar.markdown("""
1. Upload or paste medical notes  
2. Click **'🔍 Classify Notes'**  
3. View ICD-11 chapter, confidence, and chart  
""")
st.sidebar.info("⚠️ For educational use only. Not for clinical diagnosis.")

# =============================================================================
# INPUT SECTION
# =============================================================================
st.subheader("📄 Input Medical Notes")

uploaded_file = st.file_uploader("Upload a text file (.txt):", type=["txt"])
text_input = st.text_area(
    "Paste medical notes below:",
    height=200,
    placeholder="Example: Patient presents with chest pain and shortness of breath..."
)

if uploaded_file is not None:
    text_input = uploaded_file.read().decode("utf-8")

# =============================================================================
# CLASSIFICATION
# =============================================================================
if st.button("🔍 Classify Notes", use_container_width=True):
    if not text_input.strip():
        st.error("⚠️ Please enter or upload medical notes first.")
    else:
        with st.spinner("Analyzing medical notes..."):
            result = predict_category(text_input)
        category = result['category']
        confidence = result['confidence']
        top_preds = result['top_predictions']
        info = ICD_CHAPTERS.get(category, {'code': '??', 'description': 'Unknown', 'color': '#e5e7eb'})

        st.markdown(f"""
        <div style="background-color:{info['color']};padding:20px;border-radius:10px;border-left:6px solid #4f46e5;">
        <h3>📊 Primary Classification</h3>
        <h2>{category}</h2>
        <p><b>ICD-11 Chapter {info['code']}</b> — {info['description']}</p>
        <p style="color:#4f46e5;font-weight:bold;">Confidence: {confidence*100:.1f}%</p>
        </div>
        """, unsafe_allow_html=True)

        st.progress(confidence)

        # ===============================
        # BAR CHART FOR TOP 3 PREDICTIONS
        # ===============================
        st.markdown("### 📈 Top 3 Prediction Confidences")
        chart_data = {
            "ICD-11 Chapter": [pred["category"] for pred in top_preds],
            "Confidence (%)": [pred["confidence"] * 100 for pred in top_preds]
        }
        fig = px.bar(
            chart_data,
            x="Confidence (%)",
            y="ICD-11 Chapter",
            orientation="h",
            color="Confidence (%)",
            color_continuous_scale="Blues",
            range_color=[0, 100],
            title="Model Confidence by ICD-11 Chapter"
        )
        fig.update_layout(
            xaxis_title="Confidence (%)",
            yaxis_title="",
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            margin=dict(l=20, r=20, t=50, b=20),
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)

        # ===============================
        # ALTERNATIVE CLASSIFICATIONS
        # ===============================
        st.markdown("### 🔄 Alternative Classifications")
        for alt in top_preds[1:]:
            alt_info = ICD_CHAPTERS.get(alt['category'], {'code': '??', 'description': 'Unknown'})
            st.markdown(f"""
            <div style="background:#f9fafb;padding:10px;border-radius:8px;border:1px solid #ddd;margin-bottom:8px;">
                <b>{alt['category']}</b> ({alt_info['description']})  
                <div style="font-size:13px;color:#555;">Confidence: {alt['confidence']*100:.1f}%</div>
            </div>
            """, unsafe_allow_html=True)

# =============================================================================
# FOOTER
# =============================================================================
st.markdown("""
---
⚠️ **Disclaimer:** This AI model is for educational and reference use only.  
Always verify classifications with qualified healthcare professionals.  
""")
