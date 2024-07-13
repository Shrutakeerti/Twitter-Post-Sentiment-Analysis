import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
import joblib

def load_data(file_path):
    df = pd.read_csv(file_path)
    return df

def vectorize_text(df):
    vectorizer = TfidfVectorizer(max_features=5000)
    X = vectorizer.fit_transform(df['cleaned_text']).toarray()
    return X, vectorizer

def build_autoencoder(input_dim, encoding_dim=256):
    input_layer = Input(shape=(input_dim,))
    encoded = Dense(encoding_dim, activation='relu')(input_layer)
    decoded = Dense(input_dim, activation='sigmoid')(encoded)

    autoencoder = Model(input_layer, decoded)
    encoder = Model(input_layer, encoded)

    autoencoder.compile(optimizer='adam', loss='binary_crossentropy')
    return autoencoder, encoder

def train_autoencoder(X, autoencoder):
    autoencoder.fit(X, X, epochs=50, batch_size=256, shuffle=True, validation_split=0.2)

if __name__ == "__main__":
    file_path = '../data/cleaned_dataset.csv'
    df = load_data(file_path)
    X, vectorizer = vectorize_text(df)

    input_dim = X.shape[1]
    autoencoder, encoder = build_autoencoder(input_dim)

    train_autoencoder(X, autoencoder)

    autoencoder.save('../models/autoencoder_model.h5')
    encoder.save('../models/encoder_model.h5')
    joblib.dump(vectorizer, '../models/vectorizer.pkl')
    print("Autoencoder and encoder models saved to ../models/")
