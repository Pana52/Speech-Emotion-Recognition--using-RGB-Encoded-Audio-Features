from preprocessing_SAVEE import load_data

import numpy as np
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ModelCheckpoint
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from model import build_rnn_lstm_model


# Constants
DATA_PATH = "C:/Users/Pana/Desktop/Northumbria/Final Year/Individual Computing Project KV6003BNN01/datasets/SAVEE/"
# INPUT_SHAPE = (none, 1)  # MFCC
INPUT_SHAPE = (128, 1)  # MELSPEC
NUM_CLASSES = 10  # Number of emotion categories
EPOCHS = 500
BATCH_SIZE = 32
LEARNING_RATE = 0.001


def main():
    # Load and preprocess the dataset
    X_train, X_test, y_train, y_test = load_data(DATA_PATH)

    # Ensure input shape compatibility
    X_train = np.expand_dims(X_train, axis=-1)
    X_test = np.expand_dims(X_test, axis=-1)

    # Build the model
    model = build_rnn_lstm_model((X_train.shape[1], X_train.shape[2]), NUM_CLASSES)
    model.summary()

    # Compile the model
    model.compile(optimizer=Adam(learning_rate=LEARNING_RATE),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    # Callbacks
    early_stopping = EarlyStopping(monitor='val_loss', patience=50, verbose=1)
    model_checkpoint = ModelCheckpoint('best_model.h5', save_best_only=True, monitor='val_loss', mode='min')

    # Train the model
    history = model.fit(X_train, y_train, validation_data=(X_test, y_test),
                        epochs=EPOCHS, batch_size=BATCH_SIZE,
                        callbacks=[early_stopping, model_checkpoint])

    # Evaluate the model
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=2)
    print(f'Test accuracy: {test_acc}, Test loss: {test_loss}')

    # Make predictions
    y_pred = model.predict(X_test)
    y_pred_classes = np.argmax(y_pred, axis=1)

    # Print classification report
    print("Classification Report:")
    print(classification_report(y_test, y_pred_classes))

    # Plot confusion matrix
    cm = confusion_matrix(y_test, y_pred_classes)
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt="d", cmap='Blues')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix')
    plt.show()


if __name__ == '__main__':
    main()
