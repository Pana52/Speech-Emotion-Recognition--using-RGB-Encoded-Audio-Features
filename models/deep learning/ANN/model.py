# model.py
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.regularizers import l2


def build_model(input_shape):
    model = Sequential([
        Dense(128, activation='relu', kernel_regularizer=l2(0.001), input_shape=(input_shape,)),
        Dropout(0.5),
        Dense(64, activation='relu'),
        Dropout(0.5),
        Dense(32, activation='relu'),
        Dense(6, activation='softmax')  # ADJUST THE NUMBER OF CLASSES HERE!
    ])

    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model
