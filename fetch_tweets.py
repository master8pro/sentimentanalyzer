import requests
import pandas as pd
import os

BEARER_TOKEN = os.environ.get("TWITTER_BEARER_TOKEN")  # Set this in Streamlit secrets
SEARCH_QUERY = "Sabah OR PRN Sabah OR WARISAN OR GRS OR PH OR BN"
MAX_RESULTS = 50

def fetch_tweets():
    url = f"https://api.twitter.com/2/tweets/search/recent?query={SEARCH_QUERY}&max_results={MAX_RESULTS}&tweet.fields=created_at,text"
    headers = {
        "Authorization": f"Bearer {BEARER_TOKEN}"
    }

    response = requests.get(url, headers=headers)
    data = response.json()

    texts = []
    for tweet in data.get("data", []):
        texts.append(tweet["text"])

    df = pd.DataFrame(texts, columns=["text"])
    df["label"] = "neutral"  # placeholder, your model will classify this later
    df.to_csv("data.csv", index=False)

if __name__ == "__main__":
    fetch_tweets()
