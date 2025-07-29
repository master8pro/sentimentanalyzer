import streamlit as st
import tweepy
import os
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# Setup Twitter API
bearer_token = os.environ.get("TWITTER_BEARER_TOKEN")
client = tweepy.Client(bearer_token=bearer_token, wait_on_rate_limit=True)

# Streamlit UI
st.title("🇲🇾 PRN Sabah 17: Sentiment Analyzer")
party = st.text_input("Enter a party keyword (e.g., Warisan, GRS, PH, BN):", value="Warisan")

if party:
    st.write(f"Fetching tweets for: **{party}**...")

    # Fetch tweets (you can adjust max_results)
    query = f"{party} lang:ms -is:retweet"
    tweets = client.search_recent_tweets(query=query, max_results=50)

    tweet_list = []
    if tweets.data:
        for tweet in tweets.data:
            tweet_list.append(tweet.text)
    else:
        st.warning("No recent tweets found for this party.")

    # Display tweets
    if tweet_list:
        df = pd.DataFrame(tweet_list, columns=["Tweet"])

        # Sentiment mockup: You can replace this with your model later
        tfidf = TfidfVectorizer()
        X = tfidf.fit_transform(df["Tweet"])
        clf = LogisticRegression()
        # Dummy train (replace with real model later)
        clf.fit(X, [1 if i % 2 == 0 else 0 for i in range(len(df))])
        preds = clf.predict(X)

        df["Sentiment"] = ["Positive" if p == 1 else "Negative" for p in preds]

        st.dataframe(df)

        # Summary
        pos = df["Sentiment"].value_counts().get("Positive", 0)
        neg = df["Sentiment"].value_counts().get("Negative", 0)
        st.success(f"✅ Positive: {pos}")
        st.error(f"❌ Negative: {neg}")
