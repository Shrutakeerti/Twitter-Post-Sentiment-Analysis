import os

def run_preprocess():
    os.system('python scripts/preprocess.py')

def run_train_autoencoder():
    os.system('python scripts/train_autoencoder.py')

def run_train_classifier():
    os.system('python scripts/train_classifier.py')

def run_predict():
    os.system('python scripts/predict.py')

if __name__ == "__main__":
    run_preprocess()
    run_train_autoencoder()
    run_train_classifier()
    run_predict()
