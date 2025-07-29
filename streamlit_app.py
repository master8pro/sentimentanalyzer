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

# Expanded search queries (improve match chances)
queries = {
    "Warisan": 'Warisan OR #Warisan lang:ms -is:retweet',
    "GRS": 'GRS OR #GRS lang:ms -is:retweet',
    "PH": 'PH OR "Pakatan Harapan" OR #PakatanHarapan lang:ms -is:retweet',
    "BN": 'BN OR "Barisan Nasional" OR #BarisanNasional lang:ms -is:retweet'
}

if party:
    # Use expanded query if available
    query = queries.get(party, f"{party} lang:ms -is:retweet")
    st.text(f"Query used: {query}")
    st.write(f"📡 Fetching recent tweets about **{party}**...")

    try:
        # Fetch tweets (you can lower or increase max_results)
        tweets = client.search_recent_tweets(query=query, max_results=100)

        tweet_list = []
        if tweets.data:
            for tweet in tweets.data:
                tweet_list.append(tweet.text)

            st.success(f"✅ {len(tweet_list)} tweets fetched.")
        else:
            st.warning("⚠️ No recent tweets found for this party.")

        # Display and analyze if tweets exist
        if tweet_list:
            df = pd.DataFrame(tweet_list, columns=["Tweet"])

            # Sentiment mockup (replace with real model later)
            tfidf = TfidfVectorizer()
            X = tfidf.fit_transform(df["Tweet"])
            clf = LogisticRegression()
            clf.fit(X, [1 if i % 2 == 0 else 0 for i in range(len(df))])  # Dummy training
            preds = clf.predict(X)

            df["Sentiment"] = ["Positive" if p == 1 else "Negative" for p in preds]

            st.dataframe(df)

            # Summary stats
            pos = df["Sentiment"].value_counts().get("Positive", 0)
            neg = df["Sentiment"].value_counts().get("Negative", 0)

            st.success(f"🟢 Positive: {pos}")
            st.error(f"🔴 Negative: {neg}")
    except Exception as e:
        st.error(f"❌ Error fetching tweets: {str(e)}")
