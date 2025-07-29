import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import time

st.set_page_config(page_title="PRN-Sabah 17 Live Sentiment", layout="centered")

st.title("📊 PRN-Sabah 17 Sentiment Analyzer (Live Demo)")
st.markdown("This app analyzes **sentiment of live text data** to detect which party is gaining positive feedback.")

# Simulated live text source
sample_texts = [
    "Parti A sangat bagus dan membantu rakyat.",
    "Saya kecewa dengan Parti B.",
    "Parti C membuat kerja dengan sangat baik!",
    "Parti A banyak menabur janji sahaja.",
    "Saya suka manifesto Parti C.",
    "Parti B tidak menepati janji.",
    "Parti C sangat dipercayai.",
    "Parti A memang terbaik.",
    "Parti B lemah dan tidak efisien.",
    "Saya sokong Parti A sepenuhnya!"
]

# Labels: 1 = positive, 0 = negative
sample_labels = [1, 0, 1, 0, 1, 0, 1, 1, 0, 1]

# Training basic model
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(sample_texts)
model = LogisticRegression()
model.fit(X, sample_labels)

# Placeholder for live feed
st.markdown("### 📰 Live Feed & Prediction")

live_input = st.empty()
result_table = st.empty()

# Simulated live update
texts_stream = [
    "Parti A memberi bantuan yang bagus.",
    "Parti B banyak cakap tapi tiada tindakan.",
    "Saya yakin Parti C akan menang.",
    "Parti B tidak menyelesaikan masalah air.",
    "Manifesto Parti A sangat meyakinkan.",
    "Parti C bekerja keras di kawasan saya.",
]

df_results = pd.DataFrame(columns=["Text", "Sentiment", "Predicted Party"])

for i, text in enumerate(texts_stream):
    # Transform and predict
    X_new = vectorizer.transform([text])
    prediction = model.predict(X_new)[0]

    # Assume we try to guess the party from the text
    if "Parti A" in text:
        party = "Parti A"
    elif "Parti B" in text:
        party = "Parti B"
    elif "Parti C" in text:
        party = "Parti C"
    else:
        party = "Unknown"

    sentiment = "Positive" if prediction == 1 else "Negative"
    df_results.loc[len(df_results)] = [text, sentiment, party]

    # Display results
    result_table.dataframe(df_results, use_container_width=True)
    time.sleep(3)

st.success("✔ Live update completed!")
