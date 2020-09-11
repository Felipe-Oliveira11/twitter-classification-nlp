import numpy as np
import pandas as pd
import time
import pickle
import os
import io
import json
from tensorflow.keras.models import model_from_json
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences


# Parameters
EMBEDDING_GLOVE = 300
EMBEDDING_DIM = 200
MAX_SEQUENCE_LENGTH = 55
NUM_WORDS = 227664


# Compile parameters
optimizer = RMSprop(learning_rate=0.001)
loss = BinaryCrossentropy()
metrics = ['accuracy']


def tweet_predict(text):

    # Load architecture
    with open('model.json', 'r') as json_file:
        json_savedModel = json_file.read()

    model = model_from_json(json_savedModel)

    # Load Weights
    model.load_weights("model.h5")

    model.compile(optimizer=optimizer,
                  loss=loss,
                  metrics=metrics)

    # tokenizer load
    with open('tokenizer.pkl', 'rb') as handle:
        tokenizer = pickle.load(handle)

    # time
    time_pred = time.time()

    # tweet
    text = [text]

    # tokenization
    text = tokenizer.texts_to_sequences(text)

    # pad sequences
    text = pad_sequences(text, maxlen=MAX_SEQUENCE_LENGTH)

    # predict
    sentiment = model.predict(text)
    sentiment = np.argmax(sentiment, axis=1)

    return sentiment
