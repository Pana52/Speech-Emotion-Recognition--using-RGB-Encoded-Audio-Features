# train.py
import numpy as np
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ModelCheckpoint
from preprocessing import load_data
from model import build_rnn_lstm_model

# Constants
DATA_PATH = "C:/Users/Pana/Desktop/Northumbria/Final Year/Individual Computing Project KV6003BNN01/datasets/CREMAD/"
INPUT_SHAPE = (None, 1)  # Adjust based on your extracted features dimension
NUM_CLASSES = 6  # Number of emotion categories
EPOCHS = 100
BATCH_SIZE = 32
LEARNING_RATE = 0.001


def main():
    # Load and preprocess the dataset
    X_train, X_test, y_train, y_test = load_data(DATA_PATH)

    # Ensure input shape compatibility
    # This reshaping is crucial for RNN/LSTM models as they expect 3D input: [samples, timesteps, features]
    # Adjust the reshape method according to the actual shape of your preprocessed features
    X_train = np.expand_dims(X_train, axis=-1)
    X_test = np.expand_dims(X_test, axis=-1)

    # Build the model
    model = build_rnn_lstm_model((X_train.shape[1], 1), NUM_CLASSES)
    model.summary()

    # Compile the model
    model.compile(optimizer=Adam(learning_rate=LEARNING_RATE),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    # Callbacks for early stopping and model checkpointing
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, verbose=1)
    model_checkpoint = ModelCheckpoint('best_model.h5', save_best_only=True, monitor='val_loss', mode='min')

    # Train the model
    history = model.fit(X_train, y_train, validation_data=(X_test, y_test),
                        epochs=EPOCHS, batch_size=BATCH_SIZE,
                        callbacks=[early_stopping, model_checkpoint])

    # Evaluate the model
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=2)
    print(f'Test accuracy: {test_acc}, Test loss: {test_loss}')


if __name__ == '__main__':
    main()
