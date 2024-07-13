import re
import pandas as pd
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

def load_data(file_path):
    df = pd.read_csv(file_path)
    return df

def clean_text(text):
    nltk.download('stopwords')
    nltk.download('punkt')
    nltk.download('wordnet')

    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()

    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    text = re.sub(r'\@w+|\#', '', text)
    text = re.sub(r'[^\w\s]', '', text)
    text = text.lower()
    text = word_tokenize(text)
    text = [lemmatizer.lemmatize(word) for word in text if word not in stop_words]
    return ' '.join(text)

def preprocess_data(df):
    df['cleaned_text'] = df['text'].apply(clean_text)
    return df

if __name__ == "__main__":
    file_path = '../data/your_dataset.csv'
    df = load_data(file_path)
    df = preprocess_data(df)
    df.to_csv('../data/cleaned_dataset.csv', index=False)
    print("Preprocessing completed and saved to cleaned_dataset.csv")
