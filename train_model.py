import pandas as pd
import re
import nltk
import pickle
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# Download NLTK resources
nltk.download('stopwords')
nltk.download('wordnet')

# Load datasets
imdb_df = pd.read_csv("data/imdb_reviews.csv")
custom_df = pd.read_csv("data/custom_reviews.csv")
df = pd.concat([imdb_df, custom_df], ignore_index=True)

print("Dataset loaded:", df.shape)

# NLP tools
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

# Slang normalization (MUST MATCH app.py)
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

# Clean text
df['cleaned_review'] = df['review'].apply(clean_text)

# TF-IDF with n-grams
vectorizer = TfidfVectorizer(
    max_features=7000,
    ngram_range=(1, 2)
)

X = vectorizer.fit_transform(df['cleaned_review'])
y = df['sentiment']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Evaluation
y_pred = model.predict(X_test)
print("\nAccuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Save model & vectorizer
pickle.dump(model, open("model/sentiment_model.pkl", "wb"))
pickle.dump(vectorizer, open("model/vectorizer.pkl", "wb"))

print("\nModel and vectorizer saved successfully!")
