from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout, Masking, Embedding


def create_model(input_shape, num_classes):
    model = Sequential([
        # Masking layer to handle different input sequence lengths
        Masking(mask_value=0., input_shape=input_shape),
        # LSTM layer
        LSTM(64, return_sequences=False),
        Dropout(0.5),
        Dense(64, activation='relu'),
        Dropout(0.5),
        # Output layer
        Dense(num_classes, activation='softmax')
    ])

    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    return model
