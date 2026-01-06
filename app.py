import streamlit as st
import pickle
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Download NLTK data
nltk.download('stopwords')
nltk.download('wordnet')

# Load model & vectorizer
model = pickle.load(open("model/sentiment_model.pkl", "rb"))
vectorizer = pickle.load(open("model/vectorizer.pkl", "rb"))

# NLP tools
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

# Slang normalization
slang_map = {
    "shit": "bad",
    "bullshit": "bad",
    "dubshit": "bad",
    "awfull": "awful",
    "awful": "bad",
    "fuck": "",
    "fucking": "",
    "overrated": "bad",
    "fantastic": "excellent",
    "great": "excellent",
    "awesome": "excellent",
    "good": "good",
    "peak": "excellent",
    "goat": "excellent"
}

# Sentiment keywords
positive_words = {"good", "excellent", "great", "amazing", "fantastic", "peak", "goat"}
negative_words = {"bad", "awful", "worst", "shit", "overrated"}

def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z]', ' ', text)
    words = text.split()

    cleaned_words = []
    for w in words:
        if w in slang_map:
            w = slang_map[w]
        if w not in stop_words:
            cleaned_words.append(lemmatizer.lemmatize(w))

    return " ".join(cleaned_words)

# ---------------- UI ----------------

st.set_page_config(page_title="Sentiment Analysis", layout="centered")
st.title("🎭 Sentiment Analysis App")
st.write("Enter a movie review and get its sentiment.")

user_input = st.text_area("✍️ Your review here:")

if st.button("Analyze Sentiment"):
    if user_input.strip() == "":
        st.warning("Please enter some text.")
    else:
        cleaned = clean_text(user_input)

        # If text is too weak
        important_words = {"bad", "good", "excellent", "awful"}
        if len(cleaned.split()) < 2 and not any(w in cleaned.split() for w in important_words):
            st.warning("Please enter a more descriptive review.")
        else:
            vectorized = vectorizer.transform([cleaned])
            probs = model.predict_proba(vectorized)[0]
            neg_prob, pos_prob = probs[0], probs[1]

            words = set(cleaned.split())
            has_positive = len(words & positive_words) > 0
            has_negative = len(words & negative_words) > 0

            # 🔥 PRIORITY RULES
            if has_positive and not has_negative:
                st.success("😊 Positive")
            elif has_negative and not has_positive:
                st.error("😞 Negative")
            elif has_positive and has_negative:
                st.info("😐 Average")
            else:
                # fallback to ML
                if pos_prob >= 0.65:
                    st.success("😊 Positive")
                elif pos_prob <= 0.35:
                    st.error("😞 Negative")
                else:
                    st.info("😐 Average")
