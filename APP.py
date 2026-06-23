
import streamlit as st
import joblib
import pandas as pd
from urllib.parse import urlparse
import re

# =====================================================
# PAGE CONFIG
# =====================================================
st.set_page_config(
    page_title="Behavioral Phishing Detection",
    page_icon="🛡️",
    layout="centered"
)

st.title("🛡️ Behavioral Phishing Detection System")
st.markdown("Real-Time URL Analysis using Machine Learning")

# =====================================================
# LOAD MODEL FILES
# =====================================================
try:
    model = joblib.load("phishing_detector_v2.pkl")
    pca = joblib.load("pca_v2.pkl")
    scaler = joblib.load("scaler_v2.pkl")
except Exception as e:
    st.error(f"Error Loading Model Files: {e}")
    st.stop()

# =====================================================
# FEATURE EXTRACTION
# =====================================================
def extract_features(url):
    parsed = urlparse(url)

    return {
        "URLLength": len(url),
        "DomainLength": len(parsed.netloc),
        "IsHTTPS": 1 if parsed.scheme.lower() == "https" else 0,
        "NoOfSubDomain": max(parsed.netloc.count(".") - 1, 0),
        "NoOfOtherSpecialCharsInURL": len(
            re.findall(r'[!@#$%^&*(),.?":{}|<>]', url)
        ),
        "NoOfEqualsInURL": url.count("="),
        "NoOfQMarkInURL": url.count("?"),
        "NoOfAmpersandInURL": url.count("&"),
        "NoOfLettersInURL": sum(c.isalpha() for c in url)
    }

# =====================================================
# URL INPUT
# =====================================================
url_input = st.text_input(
    "🔗 Enter URL",
    placeholder="https://example.com"
)

# =====================================================
# ANALYZE BUTTON
# =====================================================
if st.button("Analyze URL"):

    if not url_input.strip():
        st.warning("Please enter a URL.")
        st.stop()

    try:

        # =============================================
        # FEATURE EXTRACTION
        # =============================================
        features = extract_features(url_input)

        sample_df = pd.DataFrame([features])

        # =============================================
        # PREPROCESSING
        # =============================================
        sample_scaled = scaler.transform(sample_df)

        sample_pca = pca.transform(sample_scaled)

        # =============================================
        # PREDICTION
        # =============================================
        prediction = model.predict(sample_pca)

        probabilities = model.predict_proba(sample_pca)

        # Class 1 = Phishing
        phishing_prob = probabilities[0][1]

        risk_score = round(phishing_prob * 100, 2)

        # =============================================
        # RESULTS
        # =============================================
        st.subheader("📊 Detection Results")

        if prediction[0] == 1:

            st.error(
                f"🚨 PHISHING DETECTED\n\nRisk Score: {risk_score}/100"
            )

            st.warning("Recommended Action: BLOCK ACCESS")

        else:

            st.success(
                f"✅ SAFE / LEGITIMATE\n\nRisk Score: {risk_score}/100"
            )

            st.info("Recommended Action: ALLOW ACCESS")

        # =============================================
        # MODEL CONFIDENCE
        # =============================================
        st.subheader("📈 Model Confidence")

        st.write(
            f"**Phishing Probability:** {phishing_prob*100:.4f}%"
        )

        st.write(
            f"**Legitimate Probability:** {(1-phishing_prob)*100:.4f}%"
        )

        # =============================================
        # BEHAVIORAL ANALYSIS
        # =============================================
        st.subheader("🧠 Behavioral Analysis Report")

        col1, col2 = st.columns(2)

        with col1:

            if len(url_input) > 75:
                st.warning("High URL Length Detected")

            if features["NoOfSubDomain"] > 2:
                st.warning("Excessive Subdomains Detected")

            if features["IsHTTPS"] == 0:
                st.warning("HTTP Connection (Not Secure)")

        with col2:

            if re.search(
                r'login|verify|bank|secure|update',
                url_input.lower()
            ):
                st.warning("Sensitive Keywords Detected")

            if features["NoOfOtherSpecialCharsInURL"] > 5:
                st.warning("High Number of Special Characters")

        # =============================================
        # FEATURE DATA
        # =============================================
        with st.expander("📋 View Extracted Features"):
            st.dataframe(sample_df)

        # =============================================
        # DEBUG INFO
        # =============================================
        with st.expander("🔧 Debug Information"):

            st.write("Prediction:", prediction)

            st.write("Classes:", model.classes_)

            st.write("Probabilities:", probabilities)

            st.write("Probability Shape:", probabilities.shape)

    except Exception as e:
        st.error(f"Prediction Error: {e}")
