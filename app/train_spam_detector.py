import numpy as np
import pandas as pd

# Read the Data

data = pd.read_csv('./data/spam_data.csv')

# Text Preprocessing

import re # regex library
def preprocessor(text):
    text = re.sub('<[^>]*>', '', text) # Effectively removes HTML markup tags
    emoticons = re.findall('(?::|;|=)(?:-)?(?:\)|\(|D|P)', text)
    text = re.sub('[\W]+', ' ', text.lower()) + ' '.join(emoticons).replace('-', '')
    return text

# Train, Test Split

from sklearn.model_selection import train_test_split
X = data['Message'].apply(preprocessor)
y = data['Category']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Training a Neural Network Pipeline

from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import cross_val_score

tfidf = TfidfVectorizer(strip_accents=None, lowercase=False, 
                        max_features=700, 
                        ngram_range=(1,1))
neural_net_pipeline = Pipeline([('vectorizer', tfidf), 
                                ('nn', MLPClassifier(hidden_layer_sizes=(700, 700)))])

neural_net_pipeline.fit(X_train, y_train)

# Testing the Pipeline

y_pred = neural_net_pipeline.predict(X_test)
print(classification_report(y_test, y_pred))
print('Accuracy: {} %'.format(100 * accuracy_score(y_test, y_pred)))

# Saving the Pipeline

from joblib import dump
dump(neural_net_pipeline, 'spam_classifier.joblib')