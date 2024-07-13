import pandas as pd
import joblib
import numpy as np

def load_models():
    autoencoder = tf.keras.models.load_model('../models/autoencoder_model.h5')
    encoder = tf.keras.models.load_model('../models/encoder_model.h5')
    vectorizer = joblib.load('../models/vectorizer.pkl')
    clf = joblib.load('../models/sentiment_classifier.pkl')
    return autoencoder, encoder, vectorizer, clf

def predict_sentiment(texts, vectorizer, encoder, clf):
    X = vectorizer.transform(texts).toarray()
    encoded_X = encoder.predict(X)
    sentiments = clf.predict(encoded_X)
    return sentiments

if __name__ == "__main__":
    autoencoder, encoder, vectorizer, clf = load_models()

    texts = ["Example tweet text here"]  # Replace with actual data
    sentiments = predict_sentiment(texts, vectorizer, encoder, clf)
    print(f"Predicted sentiments: {sentiments}")
