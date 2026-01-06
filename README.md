# 🎭 Sentiment Analysis Web App

An end-to-end **AI-powered Sentiment Analysis application** that classifies movie reviews as **Positive, Negative, or Average** using Natural Language Processing (NLP) and Machine Learning.  
The system is designed to handle **real-world informal language**, slang, profanity, and mixed sentiments, and is deployed as a live web application using **Streamlit**.

---

## 🚀 Features

- Text preprocessing using NLP techniques (tokenization, stopword removal, lemmatization)
- TF-IDF vectorization with **n-grams** for better phrase understanding
- Logistic Regression classifier
- Handles **slang, profanity, and informal language**
- Supports **mixed sentiment detection** (Average)
- Rule-based overrides for short or ambiguous inputs
- Clean and interactive web interface
- Fully deployed online using Streamlit Cloud

---

## 🧠 Sentiment Categories

| Sentiment | Description |
|---------|------------|
| 😊 Positive | Clearly favorable opinion |
| 😞 Negative | Clearly unfavorable opinion |
| 😐 Average | Mixed or neutral opinion |

---

## 🛠️ Tech Stack

- **Language:** Python  
- **Libraries:**  
  - pandas  
  - numpy  
  - scikit-learn  
  - nltk  
  - streamlit  
- **ML Model:** Logistic Regression  
- **Text Representation:** TF-IDF with unigrams & bigrams  
- **Deployment:** Streamlit Cloud  

---

## 📂 Project Structure

sentiment-analysis/
│
├── app.py # Streamlit web application
├── train_model.py # Model training pipeline
├── requirements.txt
│
├── model/
│ ├── sentiment_model.pkl
│ └── vectorizer.pkl
│
├── data/
│ ├── imdb_reviews.csv
│ └── custom_reviews.csv


---

## 🔬 How It Works

1. **Data Loading**
   - IMDb movie reviews dataset
   - Custom domain-specific reviews (slang & informal language)

2. **Text Preprocessing**
   - Lowercasing
   - Removing punctuation
   - Stopword removal
   - Lemmatization
   - Slang normalization (e.g., “shit” → “bad”, “goat” → “excellent”)

3. **Feature Extraction**
   - TF-IDF vectorization with n-grams (1,2)

4. **Model Training**
   - Logistic Regression classifier
   - Train-test split for evaluation

5. **Prediction Logic**
   - Probability-based classification
   - Rule-based overrides for:
     - Short inputs
     - Mixed sentiment reviews

6. **Deployment**
   - Streamlit-based UI
   - Hosted on Streamlit Cloud

---

## 🧪 Example Predictions

| Input | Output |
|------|-------|
| "movie is good" | 😊 Positive |
| "this movie is shit" | 😞 Negative |
| "good first half but bad climax" | 😐 Average |
| "that actor is the goat" | 😊 Positive |

---

## 📈 Model Performance

- Accuracy: **~88–90%**
- Balanced precision and recall
- Optimized for real-world user input rather than only formal text



## 📌 Future Improvements

- Add confidence visualization
- Support multilingual sentiment analysis
- Upgrade to transformer-based models (BERT)
- Store user feedback for continuous learning

---
