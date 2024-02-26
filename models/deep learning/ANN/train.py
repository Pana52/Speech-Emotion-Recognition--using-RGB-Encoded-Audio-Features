# train.py
from preprocessing_CREMAD import load_data
from model import build_model
from keras.callbacks import ModelCheckpoint, EarlyStopping
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import numpy as np


def train_and_evaluate(model, X_train, X_test, y_train, y_test):
    # Define callbacks
    checkpoint = ModelCheckpoint('best_model.h5', save_best_only=True, monitor='val_loss', mode='min')
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

    # Train the model
    history = model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=500,  # You might adjust this depending on the callback behavior
        batch_size=32,
        callbacks=[checkpoint, early_stopping]
    )

    # Evaluate the model
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=2)
    print(f'Test accuracy: {test_acc}, Test loss: {test_loss}')

    # Plot training history
    plt.figure(figsize=(8, 8))
    plt.subplot(2, 1, 1)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.legend()
    plt.title('Accuracy')

    plt.subplot(2, 1, 2)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.legend()
    plt.title('Loss')
    plt.show()

    # Confusion Matrix and Classification Report
    y_pred = model.predict(X_test)
    y_pred_classes = np.argmax(y_pred, axis=1)

    print('Classification Report')
    print(classification_report(y_test, y_pred_classes))

    cm = confusion_matrix(y_test, y_pred_classes)
    sns.heatmap(cm, annot=True)


if __name__ == "__main__":
    data_path = "C:/Users/Pana/Desktop/Northumbria/Final Year/Individual Computing Project KV6003BNN01/datasets/CREMAD/"
    X_train, X_test, y_train, y_test = load_data(data_path)
    model = build_model(X_train.shape[1])
    train_and_evaluate(model, X_train, X_test, y_train, y_test)
