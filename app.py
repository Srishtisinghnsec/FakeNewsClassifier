import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score
from tensorflow.keras.preprocessing.text import one_hot
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense
import joblib
import streamlit as st

# Download stopwords
nltk.download('stopwords')

# Global Constants
VOCAB_SIZE = 5000
SENT_LENGTH = 20
EMBEDDING_VECTOR_FEATURES = 40
MODEL_PATH = 'fake_news_classifier.pkl'
TOKENIZER_PATH = 'tokenizer.pkl'

# Preprocess a single text
def preprocess_single_text(text):
    ps = PorterStemmer()
    review = re.sub('[^a-zA-Z]', ' ', text)  # Remove non-alphabetic characters
    review = review.lower()  # Convert to lowercase
    review = review.split()  # Tokenize the words
    review = [ps.stem(word) for word in review if word not in stopwords.words('english')]  # Stemming
    return ' '.join(review)

# Preprocess the dataset
def preprocess_data(X):
    ps = PorterStemmer()
    corpus = []
    for i in range(len(X)):
        review = re.sub('[^a-zA-Z]', ' ', X[i])
        review = review.lower()
        review = review.split()
        review = [ps.stem(word) for word in review if word not in stopwords.words('english')]
        corpus.append(' '.join(review))
    return corpus

# Tokenize and pad text for training
def tokenize_and_pad_text(corpus):
    one_hot_repr = [one_hot(words, VOCAB_SIZE) for words in corpus]
    embedded_docs=pad_sequences(one_hot_repr,padding='pre',maxlen=SENT_LENGTH)


# Tokenize and pad a single input
def tokenize_and_pad_single_input(text):
    processed_text = preprocess_single_text(text)
    sequence = [one_hot(words, VOCAB_SIZE) for words in processed_text]
    return pad_sequences(sequence, maxlen=SENT_LENGTH, padding='pre')

# Build the model
def build_model():
    model = Sequential()
    model.add(Embedding(VOCAB_SIZE, EMBEDDING_VECTOR_FEATURES, input_length=SENT_LENGTH))
    model.add(LSTM(100))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    print(model.summary())
    return model

# Train the model
def train_model(X_train, y_train, X_val, y_val):
    model = build_model()
    model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=50, batch_size=50)
    joblib.dump(model, MODEL_PATH)  # Save model
    print(f"Model saved as '{MODEL_PATH}'")
    return model

# Test the model
def test_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    y_pred_binary = (y_pred > 0.5).astype(int)
    cm = confusion_matrix(y_test, y_pred_binary)
    accuracy = accuracy_score(y_test, y_pred_binary)
    print("Confusion Matrix:\n", cm)
    print("Accuracy:", accuracy)

# Predict real-time input
def predict_input(text):
    model = joblib.load(MODEL_PATH)
    padded_input = tokenize_and_pad_single_input(text)
    prediction = model.predict(padded_input)
    return "FAKE" if prediction[0][0] > 0.5 else "REAL"

# Main Streamlit Interface
def main():
    st.title("Fake News Classifier")
    st.write("Enter a news headline or text to check if it is real or fake.")

    # Input from user
    user_input = st.text_area("Enter news headline:")
    if st.button("Predict"):
        if user_input.strip():
            prediction = predict_input(user_input)
            st.write(f"Prediction: **{prediction}**")
        else:
            st.write("Please enter some text.")

if __name__ == "__main__":
    main()
