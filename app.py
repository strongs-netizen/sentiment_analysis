import streamlit as st
from transformers import pipeline

# -----------------------------
# PAGE CONFIG
# -----------------------------
st.set_page_config(
    page_title="Sentiment Analysis App",
    page_icon="🤖",
    layout="centered"
)

# -----------------------------
# LOAD MODEL
# -----------------------------
@st.cache_resource
def load_model():
    classifier = pipeline(
        "text-classification",
        model="barissayil/bert-sentiment-analysis-sst"
    )
    return classifier

pipe = load_model()

# -----------------------------
# APP UI
# -----------------------------
st.title("🤖 Sentiment Analysis App")
st.write(
    "Analyze whether a sentence has a positive or negative sentiment using a Hugging Face BERT model."
)

# User input
text = st.text_area(
    "Enter your text below:",
    height=150,
    placeholder="Type something here..."
)

# Analyze button
if st.button("Analyze Sentiment"):

    if text.strip() == "":
        st.warning("Please enter some text.")
    else:
        with st.spinner("Analyzing..."):

            result = pipe(text)

            label = result[0]["label"]
            score = result[0]["score"]

            # -----------------------------
            # DISPLAY RESULT
            # -----------------------------
            st.subheader("Result")

            if label.upper() == "POSITIVE":
                st.success(f"Sentiment: {label}")
            else:
                st.error(f"Sentiment: {label}")

            st.write(f"Confidence Score: {score:.4f}")

# -----------------------------
# SIDEBAR
# -----------------------------
st.sidebar.title("About")
st.sidebar.info(
    """
    This app uses a pretrained BERT model from Hugging Face
    for sentiment analysis.

    Model:
    barissayil/bert-sentiment-analysis-sst
    """
)