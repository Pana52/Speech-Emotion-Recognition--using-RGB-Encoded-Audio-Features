# train.py
import numpy as np
from keras.callbacks import ModelCheckpoint, EarlyStopping
from preprocessing import load_data
from model import build_resnet

# Constants
DATA_PATH = "C:/Users/Pana/Desktop/Northumbria/Final Year/Individual Computing Project KV6003BNN01/datasets/CREMAD/"
INPUT_SHAPE = (160, 1)  # Adjust this to match the feature dimension of your data
NUM_CLASSES = 6  # Number of emotion categories
EPOCHS = 50
BATCH_SIZE = 32


def main():
    # Load and preprocess the dataset
    X_train, X_test, y_train, y_test = load_data(DATA_PATH)

    # Ensure input shape compatibility
    X_train = np.expand_dims(X_train, axis=-1)
    X_test = np.expand_dims(X_test, axis=-1)

    # Build the model
    model = build_resnet(INPUT_SHAPE, NUM_CLASSES)
    model.summary()

    # Compile the model
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    # Callbacks
    checkpoint = ModelCheckpoint('best_model.h5', save_best_only=True, monitor='val_loss', mode='min')
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, verbose=1, mode='min')

    # Train the model
    history = model.fit(X_train, y_train,
                        validation_data=(X_test, y_test),
                        epochs=EPOCHS,
                        batch_size=BATCH_SIZE,
                        callbacks=[checkpoint, early_stopping])

    # Evaluate the model
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=2)
    print(f'Test accuracy: {test_acc}, Test loss: {test_loss}')


if __name__ == '__main__':
    main()
