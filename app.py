"""
PhishGuard-Transformer - Streamlit demo

Paste a URL, see both models' verdicts side by side:
  1. XGBoost (classic ML, 12 hand-picked lexical/host features)
  2. DistilBERT (fine-tuned transformer, raw URL text)

Run with: streamlit run app.py

Needs these in the same folder:
  - xgboost_phishing_model.json
  - xgboost_results.json
  - phishnet_distilbert_final/   (from Step 3)
  - distilbert_results.json
"""

import re
import json
import numpy as np
import pandas as pd
import streamlit as st
from urllib.parse import urlparse
from xgboost import XGBClassifier

# cache these so streamlit doesn't reload the models on every single click
# (it reruns the whole script on every interaction)

@st.cache_resource
def load_xgboost_model():
    model = XGBClassifier()
    model.load_model("xgboost_phishing_model.json")
    return model


@st.cache_resource
def load_distilbert_model():
    """
    Loads DistilBERT from Step 3. Returns (None, None) if the folder isn't
    there yet, so the app still runs and shows the XGBoost side.
    """
    try:
        from transformers import AutoTokenizer, TFAutoModelForSequenceClassification
        tokenizer = AutoTokenizer.from_pretrained("phishnet_distilbert_final")
        model = TFAutoModelForSequenceClassification.from_pretrained("phishnet_distilbert_final")
        return tokenizer, model
    except Exception as e:
        return None, None


@st.cache_data
def load_saved_results():
    """Loads the metrics JSON files from Steps 2 and 3, if they exist."""
    results = {}
    try:
        with open("xgboost_results.json") as f:
            results["xgboost"] = json.load(f)
    except FileNotFoundError:
        results["xgboost"] = None
    try:
        with open("distilbert_results.json") as f:
            results["distilbert"] = json.load(f)
    except FileNotFoundError:
        results["distilbert"] = None
    return results


# same feature extraction as Step 2 - has to match exactly or predictions break

def extract_features(url: str) -> pd.DataFrame:
    url = str(url)
    parsed = urlparse(url if "://" in url else "http://" + url)
    domain = parsed.netloc

    length = len(url)
    num_dots = url.count(".")
    num_hyphens = url.count("-")
    num_digits = sum(c.isdigit() for c in url)
    num_special = len(re.findall(r"[^a-zA-Z0-9.\-/:]", url))
    has_https = int(parsed.scheme == "https")
    has_ip = int(bool(re.match(r"^(\d{1,3}\.){3}\d{1,3}$", domain)))
    num_subdirs = url.count("/")
    domain_length = len(domain)
    at_symbol = int("@" in url)
    suspicious_words = ["login", "verify", "update", "secure", "account", "bank", "confirm", "signin", "webscr"]
    suspicious_count = sum(w in url.lower() for w in suspicious_words)
    probs = [url.count(c) / length for c in set(url)] if length > 0 else [0]
    entropy = -sum(p * np.log2(p) for p in probs if p > 0)

    feats = {
        "url_length": length, "num_dots": num_dots, "num_hyphens": num_hyphens,
        "num_digits": num_digits, "num_special_chars": num_special, "has_https": has_https,
        "has_ip": has_ip, "num_subdirs": num_subdirs, "domain_length": domain_length,
        "has_at_symbol": at_symbol, "suspicious_word_count": suspicious_count, "entropy": entropy
    }
    return pd.DataFrame([feats])


def predict_xgboost(model, url: str):
    feats = extract_features(url)
    pred = int(model.predict(feats)[0])
    proba = float(model.predict_proba(feats)[0][1])  # probability of class 1 (phishing)
    return pred, proba


def predict_distilbert(tokenizer, model, url: str):
    import tensorflow as tf
    encodings = tokenizer([url], padding="max_length", truncation=True, max_length=64, return_tensors="tf")
    logits = model(encodings).logits
    probs = tf.nn.softmax(logits, axis=1).numpy()[0]
    pred = int(np.argmax(probs))
    proba = float(probs[1])  # probability of class 1 (phishing)
    return pred, proba


def verdict_label(pred: int) -> str:
    return "🚨 PHISHING" if pred == 1 else "✅ Legitimate"


# ----------------------------------------------------------------------------
# APP LAYOUT
# ----------------------------------------------------------------------------

st.set_page_config(page_title="PhishGuard-Transformer", page_icon="🛡️", layout="centered")

st.title("🛡️ PhishGuard-Transformer")
st.caption("Classical ML vs. fine-tuned deep learning for phishing URL detection")

results = load_saved_results()

url_input = st.text_input("Enter a URL to check:", placeholder="https://example.com/login")
check_clicked = st.button("Check URL", type="primary")

if check_clicked and url_input.strip():
    xgb_model = load_xgboost_model()
    tokenizer, dl_model = load_distilbert_model()

    col1, col2 = st.columns(2)

    # --- Classic ML result ---
    with col1:
        st.subheader("Classic ML (XGBoost)")
        pred, proba = predict_xgboost(xgb_model, url_input)
        st.markdown(f"### {verdict_label(pred)}")
        st.metric("Phishing probability", f"{proba:.1%}")
        if results["xgboost"]:
            with st.expander("Model test-set performance"):
                r = results["xgboost"]
                st.write(f"Accuracy: {r['accuracy']:.4f}")
                st.write(f"Precision: {r['precision']:.4f}")
                st.write(f"Recall: {r['recall']:.4f}")
                st.write(f"F1: {r['f1']:.4f}")

    # --- Deep learning result ---
    with col2:
        st.subheader("Deep Learning (DistilBERT)")
        if dl_model is not None:
            pred, proba = predict_distilbert(tokenizer, dl_model, url_input)
            st.markdown(f"### {verdict_label(pred)}")
            st.metric("Phishing probability", f"{proba:.1%}")
            if results["distilbert"]:
                with st.expander("Model test-set performance"):
                    r = results["distilbert"]
                    st.write(f"Accuracy: {r['accuracy']:.4f}")
                    st.write(f"Precision: {r['precision']:.4f}")
                    st.write(f"Recall: {r['recall']:.4f}")
                    st.write(f"F1: {r['f1']:.4f}")
        else:
            st.warning(
                "DistilBERT model not found yet.\n\n"
                "Run Step 3 in Colab, then download the `phishnet_distilbert_final` "
                "folder into this same directory."
            )

    with st.expander("🔍 Extracted lexical features (for the classic ML model)"):
        st.dataframe(extract_features(url_input))

elif check_clicked:
    st.warning("Please enter a URL first.")

st.divider()
st.caption(
    "PhishGuard-Transformer — a comparative study of classical ML (XGBoost on hand-engineered "
    "lexical/host features) vs. a fine-tuned DistilBERT transformer on raw URL text, trained and "
    "evaluated on the PhiUSIIL Phishing URL Dataset."
)
