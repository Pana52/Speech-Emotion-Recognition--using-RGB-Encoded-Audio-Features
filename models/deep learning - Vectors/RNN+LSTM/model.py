# cnn_feature_extractor.py
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout, BatchNormalization, InputLayer


def build_rnn_lstm_model(input_shape, num_classes):
    model = Sequential([
        InputLayer(input_shape=input_shape),
        LSTM(128, return_sequences=True),
        BatchNormalization(),
        Dropout(0.5),
        LSTM(64),
        BatchNormalization(),
        Dropout(0.5),
        Dense(32, activation='relu'),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ])

    return model
