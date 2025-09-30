

from sklearn.base import BaseEstimator, TransformerMixin,ClassifierMixin
import re
import string
from textblob import TextBlob
import spacy
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    import subprocess
    subprocess.run(["python", "-m", "spacy", "download", "en_core_web_sm"])
    nlp = spacy.load("en_core_web_sm")



class TextPreprocessor(BaseEstimator, TransformerMixin):
    def __init__(self, chat_words):
        self.chat_words = chat_words
      

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return [self.preprocess_text(text) for text in X]

    def preprocess_text(self, text):
        # 1. Lowercase
        text = text.lower()
        
        # 2. Remove HTML tags
        text = re.sub(r'<.*?>', '', text)
        
        # 3. Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text)
        
        # 4. Remove emails
        text = re.sub(r'\S+@\S+', '', text)
        
        # 5. Remove numbers
        text = re.sub(r'\d+', '', text)
        
        # 6. Remove punctuation
        text = text.translate(str.maketrans('', '', string.punctuation))
        
        # 7. Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        # 8. Remove XXXX-like placeholders
        text = re.sub(r'\b[xX]{2,}\b', '', text)
        
        # 9. Replace chat words
        words = text.split()
        words = [self.chat_words.get(w, w) for w in words]
        text = " ".join(words)
        
        #10 auto correct using textblob
        if text.strip():
            text = str(TextBlob(text).correct())
        
        # 11. Tokenization + Stopword removal + Lemmatization
        doc = nlp(text)
        tokens=[]
        for token in doc:
        
             # Skip stopwords like "is", "the", "and"
            if token.is_stop:
                continue
        
            # Skip spaces or empty strings
            if token.is_space:
                continue
        
            # Add the lemma (base form) of the token to our list
            tokens.append(token.lemma_)
        
        return " ".join(tokens)

