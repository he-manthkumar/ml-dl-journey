import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import load_model

word_index = imdb.get_word_index()
reversed_word_index = {k:(v+3) for k,v in word_index.items()}

model = load_model('imdb_rnn_better.h5')

def decode_review(review):
    return ' '.join([reversed_word_index.get(i-3, '?') for i in review])    

def preprocess_text(text):
    words=text.lower().split()
    encoded_review = [word_index.get(word, 2) + 3 for word in words]
    padded_review = sequence.pad_sequences([encoded_review], maxlen=500)
    return padded_review

import streamlit as st

st.title('IMDB Movie Review Sentiment Analysis')
st.write('Enter a movie review to know its sentiment')

review = st.text_area('Review', 'Type Here')

if st.button('Predict Sentiment'):
    preprocess_input=preprocess_text(review)
    prediction=model.predict(preprocess_input)
    sentiment = 'Positive' if prediction[0][0] > 0.5 else 'Negative'
    st.write('Sentiment:', sentiment)
    st.write('Confidence:', prediction[0][0])
else:
    st.write('Press the above button to get the sentiment')

