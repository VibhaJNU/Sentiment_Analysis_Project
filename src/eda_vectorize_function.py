
#Import librariries

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
from collections import Counter
import pickle
import joblib   
from sklearn.feature_extraction.text import CountVectorizer


# Defined function to pre process of the dataset

def preprocess_text(text):  
    
    text = text.lower()  # Convert to lowercase
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)  # Remove URLs
    text = re.sub(r'\d+', '', text)  # Remove numbers
    stop_words = set(stopwords.words('english'))  # Get English stop words
    text = ' '.join([word for word in text.split() if word not in stop_words])  # Remove stop words
    stemmer = PorterStemmer()  # Initialize stemmer
    text = ' '.join([stemmer.stem(word) for word in text.split()])  # Apply stemming    
    lemmatizer = WordNetLemmatizer()  # Initialize lemmatizer
    text = ' '.join([lemmatizer.lemmatize(word) for word in text.split()])  # Apply lemmatization
 
    return text

# Load the most frequent words from the file
with open('most_frequent_words.pkl', 'rb') as f:
    freq_words = pickle.load(f)

# Function to remove most frequent words from the dataset
def remove_most_frequent_words(text):
    return ' '.join([word for word in text.split() if word not in freq_words])

# Function to vectorize the text using CountVectorizer
def vectorize_text(text):
    # Load the pre-trained CountVectorizer
    vectorizer = joblib.load('vectorizer.pkl')
    # Transform the text into a vector
    return vectorizer.transform([text])

