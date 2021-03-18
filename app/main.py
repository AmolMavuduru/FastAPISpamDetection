import joblib
import re
from sklearn.neural_network import MLPClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from fastapi import FastAPI

app = FastAPI()

model = joblib.load('spam_classifier.joblib')

def preprocessor(text):
    text = re.sub('<[^>]*>', '', text) # Effectively removes HTML markup tags
    emoticons = re.findall('(?::|;|=)(?:-)?(?:\)|\(|D|P)', text)
    text = re.sub('[\W]+', ' ', text.lower()) + ' '.join(emoticons).replace('-', '')
    return text

def classify_message(model, message):

	message = preprocessor(message)
	label = model.predict([message])[0]
	spam_prob = model.predict_proba([message])

	return {'label': label, 'spam_probability': spam_prob[0][1]}

@app.get('/')
def get_root():

	return {'message': 'Welcome to the spam detection API'}

@app.get('/spam_detection_query/')
async def detect_spam_query(message: str):
	return classify_message(model, message)

@app.get('/spam_detection_path/{message}')
async def detect_spam_path(message: str):
	return classify_message(model, message)