import re
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences

APP_DIR = Path(__file__).resolve().parent
MODEL_PATH = APP_DIR / "Twitter_Airline_BiLSTM.keras"
TOKENIZER_PATH = APP_DIR / "twitter_airline_tokenizer.pkl"
LABELS_PATH = APP_DIR / "twitter_airline_label_mapping.pkl"
MAX_LENGTH = 50

st.set_page_config(
    page_title="Airline Sentiment AI",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    .block-container {padding-top: 1.8rem; padding-bottom: 2.5rem; max-width: 1200px;}
    .hero {
        padding: 1.6rem 1.8rem; border-radius: 18px;
        background: linear-gradient(120deg, #10243a 0%, #174e68 58%, #287d87 100%);
        color: white; margin-bottom: 1.25rem;
    }
    .hero h1 {font-size: 2.15rem; margin: 0 0 .35rem 0; color: white;}
    .hero p {font-size: 1rem; margin: 0; color: #e6f4f7;}
    .small-muted {color: #687789; font-size: .9rem;}
    div[data-testid="stMetric"] {
        border: 1px solid rgba(128,128,128,.22); border-radius: 14px;
        padding: 1rem; background: rgba(128,128,128,.04);
    }
    .result-card {
        border: 1px solid rgba(128,128,128,.25); border-radius: 16px;
        padding: 1rem 1.2rem; margin: .4rem 0 1rem 0;
    }
    footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

def clean_tweet(text):
    text = str(text).lower()
    text = re.sub(r"https?://\S+|www\.\S+", " URL ", text)
    text = re.sub(r"@\w+", " USER ", text)
    text = re.sub(r"#(\w+)", r" \1 ", text)
    text = text.replace("&amp;", " and ")
    text = re.sub(r"[^a-z0-9'!?.,$ ]+", " ", text)
    text = re.sub(r"([!?])\1+", r" \1 ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

@st.cache_resource(show_spinner="Loading the sentiment model...")
def load_artifacts():
    missing = [p.name for p in [MODEL_PATH, TOKENIZER_PATH, LABELS_PATH] if not p.exists()]
    if missing:
        raise FileNotFoundError(
            "Missing required model artifacts: " + ", ".join(missing)
        )
    model = tf.keras.models.load_model(MODEL_PATH, compile=False)
    with open(TOKENIZER_PATH, "rb") as f:
        tokenizer = pickle.load(f)
    with open(LABELS_PATH, "rb") as f:
        mapping = pickle.load(f)
    id_to_label = mapping.get("id_to_label", {0: "negative", 1: "neutral", 2: "positive"})
    id_to_label = {int(k): str(v) for k, v in id_to_label.items()}
    return model, tokenizer, id_to_label

def predict_tweet(text, model, tokenizer, id_to_label):
    cleaned = clean_tweet(text)
    sequence = tokenizer.texts_to_sequences([cleaned])
    padded = pad_sequences(sequence, maxlen=MAX_LENGTH, padding="post", truncating="post")
    probs = model.predict(padded, verbose=0)[0]
    pred_id = int(np.argmax(probs))
    scores = {id_to_label.get(i, str(i)): float(probs[i]) for i in range(len(probs))}
    return id_to_label.get(pred_id, str(pred_id)), float(probs[pred_id]), scores

def sentiment_color(label):
    value = label.lower()
    if value == "positive":
        return "#16845b"
    if value == "negative":
        return "#c74343"
    return "#b7791f"

with st.sidebar:
    st.markdown("## ✈️ Airline Sentiment AI")
    st.caption("Bidirectional LSTM · 3-class text classification")
    st.divider()
    page = st.radio("Workspace", ["Single tweet", "Batch CSV", "About"], index=0)
    st.divider()
    st.markdown("**Classes**")
    st.markdown("🟢 Positive  \n🟡 Neutral  \n🔴 Negative")
    st.caption("Predictions are model estimates, not guaranteed facts.")

st.markdown("""
<div class="hero">
  <h1>Airline Sentiment Intelligence</h1>
  <p>Analyze customer tweets with a trained Bidirectional LSTM model.</p>
</div>
""", unsafe_allow_html=True)

try:
    model, tokenizer, id_to_label = load_artifacts()
except Exception as exc:
    st.error("The app is not ready to predict yet.")
    st.markdown("### Required files")
    st.code("Twitter_Airline_BiLSTM.keras\ntwitter_airline_tokenizer.pkl\ntwitter_airline_label_mapping.pkl")
    st.warning(
        "Only the trained .keras model was included in the uploaded files. "
        "The tokenizer and label mapping must be exported from the same training notebook "
        "so the model receives exactly the vocabulary it was trained with."
    )
    with st.expander("How to create the missing files"):
        st.markdown("""
Run the notebook section **“13. Save the Model and Tokenizer”** after training. It should create:
- `twitter_airline_tokenizer.pkl`
- `twitter_airline_label_mapping.pkl`

Put both files beside `app.py`, then rerun the app.
""")
    st.caption(f"Technical detail: {exc}")
    st.stop()

if page == "Single tweet":
    left, right = st.columns([1.08, .92], gap="large")
    with left:
        st.subheader("Analyze a tweet")
        example = st.selectbox(
            "Try an example",
            [
                "Write or paste your own tweet…",
                "The flight attendants were wonderful and very helpful!",
                "My flight was delayed again and customer service was terrible.",
                "I am flying from Boston to New York tomorrow.",
            ],
        )
        default_text = "" if example == "Write or paste your own tweet…" else example
        tweet = st.text_area(
            "Tweet text",
            value=default_text,
            height=170,
            max_chars=5000,
            placeholder="Paste an airline-related tweet here...",
            help="The app applies the same cleaning rules used in the training notebook.",
        )
        analyze = st.button("✨ Analyze sentiment", type="primary", use_container_width=True)
        st.caption(f"{len(tweet)} / 5,000 characters")
    with right:
        st.subheader("Prediction")
        if analyze:
            if not tweet.strip():
                st.warning("Enter a tweet first.")
            else:
                with st.spinner("Analyzing text..."):
                    label, confidence, scores = predict_tweet(tweet, model, tokenizer, id_to_label)
                color = sentiment_color(label)
                st.markdown(
                    f"""<div class="result-card">
                    <div class="small-muted">PREDICTED SENTIMENT</div>
                    <div style="font-size:2rem;font-weight:750;color:{color};text-transform:capitalize;">
                    {label}</div>
                    <div style="margin-top:.4rem;">Top-class probability: <b>{confidence:.1%}</b></div>
                    </div>""",
                    unsafe_allow_html=True,
                )
                st.progress(min(max(confidence, 0.0), 1.0), text="Top-class probability")
                chart_df = pd.DataFrame(
                    {"Sentiment": list(scores.keys()), "Probability": list(scores.values())}
                ).sort_values("Probability", ascending=True)
                st.bar_chart(chart_df.set_index("Sentiment"), x_label="Sentiment", y_label="Probability")
                with st.expander("View exact class probabilities"):
                    st.dataframe(
                        pd.DataFrame([
                            {"Sentiment": k.capitalize(), "Probability": f"{v:.2%}"}
                            for k, v in scores.items()
                        ]), hide_index=True, use_container_width=True
                    )
                with st.expander("Preprocessed text"):
                    st.code(clean_tweet(tweet) or "(empty after cleaning)")
        else:
            st.info("Your prediction and class probabilities will appear here.")
            st.markdown("""
            **Workflow**
            1. Paste a customer tweet.
            2. Select **Analyze sentiment**.
            3. Review the predicted class and probability scores.
            """)

elif page == "Batch CSV":
    st.subheader("Batch analysis")
    st.write("Upload a CSV containing a text column. The app adds sentiment and probability columns.")
    uploaded = st.file_uploader("Choose a CSV file", type=["csv"])
    if uploaded:
        try:
            df = pd.read_csv(uploaded)
            if df.empty:
                st.warning("The uploaded CSV is empty.")
            else:
                candidates = [c for c in df.columns if df[c].dtype == "object"]
                if not candidates:
                    st.error("No text columns were found in this CSV.")
                else:
                    text_col = st.selectbox("Select the tweet/text column", candidates)
                    if st.button("Run batch analysis", type="primary"):
                        with st.spinner("Analyzing rows..."):
                            rows = [
                                predict_tweet("" if pd.isna(x) else str(x), model, tokenizer, id_to_label)
                                for x in df[text_col].tolist()
                            ]
                        df["predicted_sentiment"] = [r[0] for r in rows]
                        df["top_probability"] = [r[1] for r in rows]
                        for label in sorted(set(k for _, _, d in rows for k in d)):
                            df[f"probability_{label}"] = [r[2].get(label, 0.0) for r in rows]
                        st.success(f"Finished analyzing {len(df):,} rows.")
                        st.dataframe(df.head(100), use_container_width=True, hide_index=True)
                        st.download_button(
                            "⬇️ Download predictions CSV",
                            data=df.to_csv(index=False).encode("utf-8-sig"),
                            file_name="airline_sentiment_predictions.csv",
                            mime="text/csv",
                            type="primary",
                        )
        except Exception as exc:
            st.error(f"Could not read this CSV: {exc}")
    with st.expander("Expected CSV format"):
        st.code("tweet\n\"The flight was excellent!\"\n\"My flight was delayed again.\"\n\"Flying tomorrow.\"")

else:
    st.subheader("About this project")
    st.markdown("""
    This dashboard serves a **three-class Twitter sentiment model** trained on airline-related tweets.

    **Model:** Embedding → SpatialDropout1D → Bidirectional LSTM → Dense classifier  
    **Labels:** Negative, Neutral, Positive  
    **Input preparation:** URL and user-mention normalization, hashtag-word preservation, consistent tokenization, and post-padding to 50 tokens.

    **Important:** the displayed probability is the model's confidence-like output, not a guarantee that a prediction is correct. For academic reporting, evaluate the model on a held-out test set and report macro F1 alongside the classification report.
    """)
    st.info("The tokenizer and label mapping must come from the same training run as the model.")

st.divider()
st.caption("Airline Sentiment AI · Streamlit deployment package")
