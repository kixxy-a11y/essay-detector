import streamlit as st
import joblib
import os

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="AI vs Human Detector", page_icon="📝")

# --- LOAD MODELS ---
@st.cache_resource # This keeps the model in RAM so it's fast
def load_models():
    model = joblib.load('models/detector_model.pkl')
    vectorizer = joblib.load('models/vectorizer.pkl')
    return model, vectorizer

try:
    model, vectorizer = load_models()
except FileNotFoundError:
    st.error("❌ Models not found! Please run 'python train.py' first.")
    st.stop()

# --- UI DESIGN ---
st.title("🤖 AI vs. ✍️ Human Essay Detector")
st.markdown("""
    Welcome to the **QCU AI Research Lab** prototype. 
    Paste an essay below to analyze its linguistic patterns.
""")

# Input Area
essay_text = st.text_area("Paste your essay here:", height=250, placeholder="Once upon a time...")

# Analyze Button
if st.button("Analyze Essay"):
    if essay_text.strip():
        # 1. Transform text
        features = vectorizer.transform([essay_text])
        
        # 2. Predict
        prediction = model.predict(features)[0]
        probability = model.predict_proba(features)[0]
        
        # 3. Display Results
        st.divider()
        if prediction == 1:
            st.error(f"### 🚩 Result: AI GENERATED")
            st.progress(probability[1])
            st.write(f"Confidence: **{probability[1]:.2%}**")
        else:
            st.success(f"### ✅ Result: HUMAN WRITTEN")
            st.progress(probability[0])
            st.write(f"Confidence: **{probability[0]:.2%}**")
            
        # Optional: Add a "Metric" view
        col1, col2 = st.columns(2)
        col1.metric("AI Score", f"{probability[1]:.1%}")
        col2.metric("Human Score", f"{probability[0]:.1%}")
    else:
        st.warning("⚠️ Please paste some text first!")

# --- FOOTER ---
st.sidebar.info("Developed for BSIT AI Practice - 2026")