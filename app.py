import streamlit as st
import pickle
import google.generativeai as genai

st.set_page_config(page_title="Fake News Detector", page_icon="📰", layout="wide")

st.markdown("""
<style>
.title-text {
    font-size: 40px; font-weight: 800;
    background: linear-gradient(90deg, #f953c6, #b91d73);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    text-align: center;
}
</style>""", unsafe_allow_html=True)

st.markdown('<div class="title-text">📰 Fake News Detector</div>', unsafe_allow_html=True)
st.markdown('<p style="text-align:center;color:#aaa;">Powered by TF-IDF + Logistic Regression + Gemini AI</p>', unsafe_allow_html=True)
st.markdown("---")

@st.cache_resource
def load_model():
    model = pickle.load(open('model/lr_model.pkl', 'rb'))
    tfidf = pickle.load(open('model/tfidf.pkl', 'rb'))
    return model, tfidf

model, tfidf = load_model()

tab1, tab2 = st.tabs(["🔍 Detect News", "ℹ️ About"])

with tab1:
    st.markdown("### Enter News Article")
    title = st.text_input("📰 News Title", placeholder="Enter the news headline...")
    content = st.text_area("📝 News Content", placeholder="Paste the news article here...", height=200)
    gemini_key = st.text_input("🔑 Gemini API Key (optional - for AI explanation)", type="password")

    if st.button("🔍 Detect", type="primary"):
        if not title and not content:
            st.warning("Please enter a title or content!")
        else:
            text = title + ' ' + content
            text_tfidf = tfidf.transform([text])
            pred = model.predict(text_tfidf)[0]
            prob = model.predict_proba(text_tfidf)[0]

            real_prob = prob[1] * 100
            fake_prob = prob[0] * 100

            c1, c2 = st.columns(2)
            with c1:
                if pred == 1:
                    st.markdown(f"""
                    <div style='background:#4ade8022;border:2px solid #4ade80;border-radius:12px;padding:20px;text-align:center'>
                        <h2 style='color:#4ade80'>✅ REAL NEWS</h2>
                        <h1 style='color:#4ade80'>{real_prob:.1f}%</h1>
                        <p style='color:#aaa'>Confidence</p>
                    </div>""", unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div style='background:#f8717122;border:2px solid #f87171;border-radius:12px;padding:20px;text-align:center'>
                        <h2 style='color:#f87171'>⚠️ FAKE NEWS</h2>
                        <h1 style='color:#f87171'>{fake_prob:.1f}%</h1>
                        <p style='color:#aaa'>Confidence</p>
                    </div>""", unsafe_allow_html=True)
            with c2:
                st.markdown("#### Probability Breakdown")
                st.metric("✅ Real", f"{real_prob:.1f}%")
                st.metric("⚠️ Fake", f"{fake_prob:.1f}%")
                st.progress(real_prob/100)

            # Gemini explanation
            if gemini_key:
                try:
                    genai.configure(api_key=gemini_key)
                    gemini = genai.GenerativeModel('gemini-2.0-flash')
                    prompt = f"""
                    A news article was classified as {'REAL' if pred==1 else 'FAKE'} with {real_prob if pred==1 else fake_prob:.1f}% confidence.
                    
                    Title: {title}
                    Content (first 500 chars): {content[:500]}
                    
                    In 3-4 sentences, explain why this news might be {'real' if pred==1 else 'fake'}. 
                    Look for signs like sensational language, missing sources, emotional manipulation, etc.
                    Keep it simple and clear.
                    """
                    response = gemini.generate_content(prompt)
                    st.markdown("### 🤖 AI Explanation (Gemini)")
                    st.info(response.text)
                except Exception as e:
                    st.warning(f"Gemini API error: {e}")

with tab2:
    st.markdown("""
    ### About This App
    This app uses Machine Learning to detect fake news articles.
    
    **How it works:**
    - Text is converted to numerical features using TF-IDF (50,000 features)
    - Logistic Regression classifier predicts Real vs Fake
    - Optional: Gemini AI explains the prediction in plain English
    
    **Dataset:** 44,000+ news articles (Kaggle)
    
    **Model Accuracy:** ~98%
    
    **Built by:** Phani Harika Soma  
    B.Tech CSE-DS | Sridevi Women's Engineering College
    """)