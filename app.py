import numpy as np
import pandas as pd
import streamlit as st
import tensorflow as tf
from inference import tweet_predict


print('Pandas {}'.format(pd.__version__))
print('Numpy {}'.format(np.__version__))
print('Streamlit {}'.format(st.__version__))
print('Tensorflow {}'.format(tf.__version__))


# main menu
def menu():

    st.sidebar.header('Home')
    page = st.sidebar.radio("", ('Twitter',
                                 'Deep learning application',
                                 'Tweet sentiment',
                                 'Contact'))
    hide_streamlit_style = """
    <style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    </style>

    """
    st.markdown(hide_streamlit_style, unsafe_allow_html=True)

    if page == 'Twitter':
        twitter()

    if page == 'Deep learning application':
        plataform()

    if page == 'Tweet sentiment':
        sentiment()

    if page == 'Contact':
        contact()


# Twitter page
def twitter():
    st.title('Twitter')
    st.write('')
    st.image(
        'https://lh3.googleusercontent.com/cZRzJN9uqUQpPtZ4SfLZm_QVI07creZ9-My0K2j65FKYH34SmD9rJ3frvK0M1a6XmMk',
        width=500,
        height=500)


# Twitter Deep learning
def plataform():
    st.title('Deep learning Application')
    st.write('This application is builded using Deep learning model and main objective is to be able that The Twitter obtain informations \
        about sentiments their users, in real-time making inference tweets in this plataform')
    st.write('\n')
    st.image('https://neilpatel.com/wp-content/uploads/2016/07/twitter.jpg',
             width=600,
             height=300)
    st.write('\n')
    st.write('The model that´s running in backend this application it has architecture is LSTM with Embeddings, The model was trained through GPU on Google Colab, in dataset where contains over 1.6 millions of tweets, for most informations about this solution: https://github.com/Felipe-Oliveira11/twitter-sentiments-nlp')


# Sentiment Prediction
def sentiment():
    st.title(
        'Tweet sentiment')
    st.image(
        'https://portalcbncampinas.com.br/wp-content/uploads/2019/11/portalcbncampinas.com.br-twittereapolitica-twitter-logo.png',
        width=500,
        height=200)
    st.write('\n')
    st.write('\n')
    st.write('\n')

    # insert tweet
    tweet = st.text_input('Insert Tweet', max_chars=280)
    if st.button('Predict Sentiment'):
        prediction = tweet_predict(tweet)
        prediction = np.argmax(prediction, axis=1)

        if prediction >= 0.50:
            st.success('Tweet sentiment: Positive')
        else:
            st.success('Tweet sentiment: Negative')


# contato
def contact():
    st.title('Contact')
    st.image('https://encrypted-tbn0.gstatic.com/images?q=tbn%3AANd9GcSh8_BbTxZSsHWdLsSVjvVGVjASl3WynpmbMg&usqp=CAU',
             width=100, height=100)
    st.write('\n')
    st.write('\n')
    st.write('This project was develop by Felipe Oliveira, \
             Questions or suggestions send me a message in e-mail or LinkedIn.')
    st.write('\n')
    st.markdown(
        '[LinkedIn](https://www.linkedin.com/in/felipe-oliveira-18a573189/)')
    st.write('\n')
    st.write('E-mail: felipe.oliveiras2000@gmail.com')


if __name__ == '__main__':
    menu()
