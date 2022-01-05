from tensorflow.keras.preprocessing.text import tokenizer_from_json
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding, LSTM, Conv1D, MaxPool1D, Dropout
from tensorflow.keras.layers import Bidirectional
from tensorflow import keras

def predict(text_df):
    X = text_df['text']
    tokenizer_js = ""
    f = open("tokenizer.json", "r")
    tokenizer_js = f.read()
    
    tokenizer = tokenizer_from_json(tokenizer_js)
    X = tokenizer.texts_to_sequences(X)
    
    maxlen = 500
    X = pad_sequences(X, maxlen=maxlen)
    
    model = keras.models.load_model('LSTM_model.h5')
    return model.predict(X)
    