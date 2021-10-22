import streamlit as st
import numpy as np
import pandas as pd
from PIL import Image
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from bs4 import BeautifulSoup
import re
from sklearn.model_selection import train_test_split

st.header('NLP Sentiment Analysis using Word2Vec and LSTM')

'''
Hi. My program can tell if you liked or hated a show on IMDb based on the review you gave. By training my model
with 25,000 reviews labeled with either negative (<5) or positive ratings, my model can now read the review you
wrote and interpret correctly (90% of the time) whether your rating for that movie/show will be positive or
negative.

Data source: https://www.kaggle.com/c/word2vec-nlp-tutorial/data

'''

image = Image.open('Photo1.PNG')
st.image(image, caption='IMDb User Review section for Squid Game')


def load_model():
    model = tf.keras.models.load_model('movie_review.h5')
    return model
model = load_model()
print(model.summary())

def clean_up(review):
    remove_html = BeautifulSoup(review,'html.parser').get_text()

    #remove punctuation and numbers
    words_only = re.sub(r'[^A-Za-z\']+',' ',remove_html)

    lower_words = words_only.lower()

    return lower_words

df_train = pd.read_csv('labeledTrainData.tsv.zip', sep='\t')
df_train['review'] = df_train['review'].apply(clean_up)
y = df_train['sentiment'].values
x = df_train['review'].values
x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.2, random_state=0)
tokenizer = Tokenizer()
tokenizer.fit_on_texts(x_train)

'''
Feel free to try it out yourself with the sample review (negative rating) below:

    "I've watched the whole lot and it's just plain silly. The cast bring over-acting to a new level,
    all the characters are spectacularly annoying (I wanted them all to die by Episode 2) and I don't
    know why I watched it all. I'm sure there were far better things I could have been doing with my life.
    Like cleaning my bathroom or cutting my toe nails. It's really woeful."

Alternatively you can try it out with user reviews of any show/movie on IMDb.com:

E.g. https://www.imdb.com/title/tt10919420/reviews

'''

input_review = st.text_area("Insert your movie/show review here and Ctrl+Enter:")

if input_review:
    with st.spinner("Analyzing your review......."):
        d = {1:input_review}
        pred = np.array(pd.Series(data=d).apply(clean_up))
        seq_pred = tokenizer.texts_to_sequences(pred)
        pad_xpred = pad_sequences(seq_pred, maxlen=4000)
        prediction = model.predict(pad_xpred)
        prediction = np.round(prediction)
        if prediction == 1:
            st.subheader('Your review is POSITIVE')
        elif prediction == 0:
            st.subheader('Your review is NEGATIVE')
