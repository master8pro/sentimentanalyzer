import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline

# Sample Malay sentiment dataset
data = {
    'text': [
        'Saya suka calon ini, sangat berwibawa dan mesra rakyat.',
        'Manifesto mereka sangat mengelirukan dan tidak realistik.',
        'Rasanya kerajaan sekarang tidak membantu rakyat.',
        'Calon wanita ini sangat pintar dan boleh dipercayai.',
        'Saya tidak yakin dengan janji politik mereka.',
        'Pencapaian mereka sangat membanggakan!',
        'Manifesto itu sangat bagus dan boleh dilaksanakan.',
        'Janji tinggal janji, tiada bukti pun.',
        'Rakyat perlukan perubahan segera.',
        'Saya tidak tahu siapa yang patut saya pilih.'
    ],
    'label': ['positive', 'negative', 'negative', 'positive', 'negative',
              'positive', 'positive', 'negative', 'neutral', 'neutral']
}
df = pd.DataFrame(data)

# Train simple model (TF-IDF + Naive Bayes)
model = make_pipeline(TfidfVectorizer(), MultinomialNB())
model.fit(df['text'], df['label'])

# Streamlit UI
st.set_page_config(page_title="PRN Sabah 17 Sentiment Analyzer", layout="centered")
st.title("🗳️ PRN Sabah 17 Sentiment Analyzer")
st.markdown("Analyze Malay political sentiment from speeches, tweets, or public comments.")

user_input = st.text_area("✍️ Masukkan ayat berkaitan PRN Sabah 17", height=150)

if st.button("🔍 Analisis Sentimen"):
    if user_input.strip() == "":
        st.warning("Sila masukkan ayat terlebih dahulu.")
    else:
        prediction = model.predict([user_input])[0]
        st.success(f"Hasil Sentimen: **{prediction.upper()}**")

st.markdown("---")
st.caption("Dibangunkan oleh Mohd Arfi | Guna model Naive Bayes + TF-IDF")
