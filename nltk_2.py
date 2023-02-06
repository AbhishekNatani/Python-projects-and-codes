import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from pprint import pprint
import streamlit as st
import pandas as pd
nltk.download([
  "names",
  "stopwords",
  "state_union",
  "twitter_samples",
  "movie_reviews",
  "vader_lexicon"
 ])
 
st.title ("Sentiment analysis framework")
st.header("type your text below")
#st.selectbox('option to type ypur statement',['yes','no'])
a = SentimentIntensityAnalyzer()

with st.expander('sentimental analysis'):
    b = st.text_input("enter your text here:")
    stopwords = nltk.corpus.stopwords.words("english")
    stopwords.extend([w.lower() for w in nltk.corpus.names.words()])
    words = [w for w in b if w.lower() not in stopwords]
    #words: list[str.isalpha(words)] = nltk.word_tokenize(b)
    words: list[str()] = nltk.word_tokenize(b)
    fd = nltk.FreqDist(words)
    #print(fd)
    #d = fd.tabulate(3)
    c = a.polarity_scores(b)
    st.write(fd)
    st.write(c)
    st.dataframe(words , 300 , 500)

    #print(c)



