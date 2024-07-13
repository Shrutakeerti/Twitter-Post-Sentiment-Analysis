import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib

def load_data(file_path):
    df = pd.read_csv(file_path)
    return df

def train_classifier(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    clf = LogisticRegression()
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    print(classification_report(y_test, y_pred))
    return clf

if __name__ == "__main__":
    file_path = '../data/labeled_dataset.csv'  # Assuming you have manually labeled this subset
    df = load_data(file_path)
    
    X = df['cleaned_text']
    y = df['sentiment']  # Assuming this column contains the sentiment labels

    vectorizer = joblib.load('../models/vectorizer.pkl')
    X = vectorizer.transform(X).toarray()

    clf = train_classifier(X, y)
    joblib.dump(clf, '../models/sentiment_classifier.pkl')
    print("Sentiment classifier saved to ../models/")
