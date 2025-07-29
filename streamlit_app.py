import streamlit as st
import tweepy
import os
import pandas as pd
from transformers import pipeline
import torch

# Load real sentiment analysis model (only once)
@st.cache_resource
def load_sentiment_model():
    return pipeline("sentiment-analysis", model="cardiffnlp/twitter-xlm-roberta-base-sentiment")
analyzer = load_sentiment_model()

# Setup Twitter API
bearer_token = os.environ.get("TWITTER_BEARER_TOKEN")
client = tweepy.Client(bearer_token=bearer_token, wait_on_rate_limit=True)

# Streamlit UI
st.title("🇲🇾 PRN Sabah 17: Real-Time Sentiment Analyzer")

party = st.text_input("Enter a party keyword (e.g., Warisan, GRS, PH, BN):", value="Warisan")

# Expanded search queries
queries = {
    "Warisan": 'Warisan OR #Warisan lang:ms -is:retweet',
    "GRS": 'GRS OR #GRS lang:ms -is:retweet',
    "PH": 'PH OR "Pakatan Harapan" OR #PakatanHarapan lang:ms -is:retweet',
    "BN": 'BN OR "Barisan Nasional" OR #BarisanNasional lang:ms -is:retweet'
}

if party:
    query = queries.get(party, f"{party} lang:ms -is:retweet")
    st.text(f"Query used: {query}")
    st.write(f"📡 Fetching recent tweets about **{party}**...")

    try:
        tweets = client.search_recent_tweets(query=query, max_results=50)
        tweet_list = [tweet.text for tweet in tweets.data] if tweets.data else []

        if not tweet_list:
            st.warning("⚠️ No recent tweets found for this party.")
        else:
            st.success(f"✅ {len(tweet_list)} tweets fetched.")

            # Run real sentiment analysis
            results = analyzer(tweet_list)

            df = pd.DataFrame({
                "Tweet": tweet_list,
                "Sentiment": [r['label'].capitalize() for r in results],
                "Score": [round(r['score'], 3) for r in results]
            })

            st.dataframe(df)

            # Summary
            summary = df["Sentiment"].value_counts()
            for label in ["Positive", "Negative", "Neutral"]:
                count = summary.get(label, 0)
                if label == "Positive":
                    st.success(f"🟢 {label}: {count}")
                elif label == "Negative":
                    st.error(f"🔴 {label}: {count}")
                else:
                    st.info(f"🔵 {label}: {count}")

    except Exception as e:
        st.error(f"❌ Error fetching tweets: {str(e)}")
