from tensorflow.keras.layers import Dense, Input
from tensorflow import keras
import tensorflow as tf
from tensorflow.keras.layers import LSTM, Dropout, GRU, Bidirectional
from tensorflow.keras.layers import GlobalAveragePooling1D
from tensorflow.keras.layers import Conv1D, SpatialDropout1D
from tensorflow.keras.layers import Embedding, Dropout
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.optimizers import Adam, RMSprop, SGD
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.models import save_model, load_model
from tensorflow.keras.models import Sequential
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from nltk.tokenize import sent_tokenize
from nltk.tokenize import word_tokenize
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.stem import SnowballStemmer
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
import nltk
import re
import string
import random
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import emoji
from PIL import Image

# %matplotlib inline
import warnings
warnings.filterwarnings('ignore')


nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')


# set global seed
seed = 42
np.random.seed(seed)
tf.random.set_seed(seed)


# loading data
path = 'training.1600000.processed.noemoticon.csv'
data = pd.read_csv(path, encoding='latin', header=None)

data.columns = ['sentiment', 'id', 'date', 'query', 'user_id', 'text']
data.drop(['id', 'date', 'query', 'user_id'], axis=1, inplace=True)
data['sentiment'] = data['sentiment'].astype(str)
data['text'] = data['text'].astype(str)

# Cleaning text


def cleaning_text(text):
    """
    Cleaning text in Twetts 
    Removing unwanted characters and emojis

                                            """

    # Removing characters and emojis
    removing_list = "@\S+|https?:\S+|http?:\S|[^A-Za-z0-9]+"
    emoji_pattern = re.compile("["
                               u"\U0001F600-\U0001F64F"  # emoticons
                               u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                               u"\U0001F680-\U0001F6FF"  # transport & map symbols
                               u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                               u"\U0001F1F2-\U0001F1F4"  # Macau flag
                               u"\U0001F1E6-\U0001F1FF"  # flags
                               u"\U0001F600-\U0001F64F"
                               u"\U00002702-\U000027B0"
                               u"\U000024C2-\U0001F251"
                               u"\U0001f926-\U0001f937"
                               u"\U0001F1F2"
                               u"\U0001F1F4"
                               u"\U0001F620"
                               u"\u200d"
                               u"\u2640-\u2642"
                               "]+", flags=re.UNICODE)

    text = emoji_pattern.sub(r'', str(text))
    text = re.sub(removing_list, " ", text)
    text = re.sub(r'\W+', ' ', text)
    text = re.sub("'", '', text)
    text = text.lower().strip()

    # Stemming and Stopwords
    stemmer = SnowballStemmer('english')
    stop_words = set(stopwords.words('english'))

    tokens = []
    for token in text.split():
        if token not in stop_words:
            tokens.append(stemmer.stem(token))
        else:
            pass

    return " ".join(tokens)


# mapping tweets sentiments
dicio = {'0': 'Negative', '4': 'Positive'}
data['sentiment'] = data['sentiment'].map(dicio)


def feature_engineering(data):
    """ Pipeline of Feature engineering for 
         text classification problem 

        1 - cleaning text 
        2 - spliting data
        3 - label encoder 
        4 - tokenization 
        5 - pad_sequences 
                                          """

    # cleaning text
    data['text'] = data['text'].apply(lambda x: cleaning_text(x))

    # spliting data
    X = data['text']
    y = data['sentiment']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.30, random_state=42)

    # LabelEncoder
    encoder = LabelEncoder()
    y_train = encoder.fit_transform(y_train.to_list())
    y_test = encoder.fit_transform(y_test.to_list())

    y_train = y_train.reshape(-1, 1)
    y_test = y_test.reshape(-1, 1)

    print('Classes: ', encoder.inverse_transform([0, 1]))
    print('\n')

    # Tokenization
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(X_train)

    # Vocabulary Size
    word_index = tokenizer.word_index
    num_words = len(word_index) + 1

    # Tokens
    sequence_train = tokenizer.texts_to_sequences(X_train)
    sequence_test = tokenizer.texts_to_sequences(X_test)

    # max sequence
    MAX_SEQUENCE_LENGTH = 55

    # pad_sequences
    X_train = pad_sequences(sequences=sequence_train,
                            maxlen=MAX_SEQUENCE_LENGTH, padding='post', truncating='post')
    X_test = pad_sequences(sequences=sequence_test,
                           maxlen=MAX_SEQUENCE_LENGTH, padding='post', truncating='post')

    return (X_train, X_test, y_train, y_test, num_words, word_index, tokenizer)


X_train, X_test, y_train, y_test, num_words, word_index, tokenizer = feature_engineering(
    data)

print(X_train.shape, y_train.shape)
print(X_test.shape, y_test.shape)

# compile parameters
optimizer = RMSprop(learning_rate=0.001)
loss = BinaryCrossentropy()
metrics = ['accuracy']


# Embedding parameters
EMBEDDING_GLOVE = 300
WORD_INDEX = word_index
EMBEDDING_DIM = 200
MAX_SEQUENCE_LENGTH = 55
NUM_WORDS = 227664


# Build LSTM
def lstm_architecture(pre_trained=True, num_words=NUM_WORDS, embedding_dim=EMBEDDING_GLOVE, max_sequence_length=MAX_SEQUENCE_LENGTH):
    """The LSTM model architecture
    with option for use Glove embedding """

    if pre_trained:

        # Glove Embedding
        GLOVE_EMB = '/content/glove.6B.300d.txt'

        embeddings_index = {}
        f = open(GLOVE_EMB)
        for line in f:
            values = line.split()
            word = value = values[0]
            coefs = np.asarray(values[1:], dtype='float32')
            embeddings_index[word] = coefs
        f.close()

        print('Found {} word vectors.'.format(len(embeddings_index)))

        # Embedding matrix
        embedding_matrix = np.zeros((num_words, embedding_dim))

        for word, i in word_index.items():
            embedding_vector = embeddings_index.get(word)
            if embedding_vector is not None:
                embedding_matrix[i] = embedding_vector

        # LSTM with Glove
        model = Sequential()
        model.add(Input(shape=max_sequence_length))
        model.add(Embedding(input_dim=num_words,
                            output_dim=embedding_dim,
                            weights=[embedding_matrix],
                            input_length=max_sequence_length,
                            trainable=False))
        model.add(SpatialDropout1D(0.20))
        model.add(Conv1D(128, kernel_size=5, strides=1,
                         padding='same', activation='relu'))
        model.add(LSTM(units=128, recurrent_dropout=0.20, return_sequences=True))
        model.add(LSTM(units=128, recurrent_dropout=0.20, return_sequences=True))
        model.add(SpatialDropout1D(0.20))
        model.add(Dense(units=512, activation='relu'))
        model.add(Dropout(0.20))
        model.add(Dense(units=64, activation='relu'))
        model.add(Dense(units=1, activation='sigmoid'))

        return model

    else:

        # LSTM without Glove
        model = Sequential()
        model.add(Input(shape=max_sequence_length))
        model.add(Embedding(input_dim=num_words,
                            output_dim=embedding_dim,
                            input_length=max_sequence_length))
        model.add(SpatialDropout1D(0.20))
        model.add(Conv1D(128, kernel_size=5, strides=1,
                         padding='same', activation='relu'))
        model.add(LSTM(units=128, recurrent_dropout=0.20, return_sequences=True))
        model.add(LSTM(units=128, recurrent_dropout=0.20, return_sequences=True))
        model.add(SpatialDropout1D(0.20))
        model.add(Dense(units=512, activation='relu'))
        model.add(Dropout(0.20))
        model.add(Dense(units=64, activation='relu'))
        model.add(Dense(units=1, activation='sigmoid'))

        return model


model = lstm_architecture(pre_trained=False, embedding_dim=EMBEDDING_DIM)

model.compile(optimizer=optimizer,
              loss=loss,
              metrics=metrics)

model.summary()

# callbacks

checkpoint = ModelCheckpoint(filepath='model.h5',
                             monitos='val_loss',
                             verbose=1,
                             save_only_weights=True)


early_stopping = EarlyStopping(monitor='val_loss',
                               min_delta=0.00001,
                               patience=10)


reduce_learning_rate = ReduceLROnPlateau(monitor='val_loss',
                                         factor=0.2,
                                         patience=5,
                                         min_delta=0.0001)


callbacks = [checkpoint, early_stopping, reduce_learning_rate]

# train model
model.fit(X_train, y_train,
          batch_size=1024,
          epochs=30,
          validation_data=(X_test, y_test),
          callbacks=[callbacks])


# save model
model.save_weights('model.h5')
