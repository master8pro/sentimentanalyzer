import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import time
import os

st.set_page_config(page_title="PRN Sabah 17 Sentiment Live Analyzer", layout="wide")

st.title("📊 PRN Sabah 17 Sentiment Analyzer (LIVE)")
data_file = 'data.csv'  # You must have this file in your repo

# Refresh every 15 seconds
refresh_interval = 15  # seconds

@st.cache_data(ttl=refresh_interval)
def load_data():
    df = pd.read_csv(data_file)
    return df

df = load_data()

# Vectorize and model
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df['text'])
model = LogisticRegression()
model.fit(X, df['label'])

# Predict live
df['prediction'] = model.predict(X)

# Display sentiment counts
sentiment_counts = df['prediction'].value_counts()
st.bar_chart(sentiment_counts)

# Show raw data if needed
if st.checkbox("Show raw data"):
    st.write(df)

st.info(f"🔄 Auto-updating every {refresh_interval} seconds...")
