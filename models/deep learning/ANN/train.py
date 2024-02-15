# train.py
import numpy as np
from preprocessing import load_data
from model import build_model
from keras.callbacks import ModelCheckpoint
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns


def train_and_evaluate(model, X_train, X_test, y_train, y_test):
    checkpoint = ModelCheckpoint('best_model.h5', save_best_only=True, monitor='val_loss', mode='min')
    history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=100, batch_size=256,
                        callbacks=[checkpoint])

    # Evaluate the model
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=2)
    print(f'Test accuracy: {test_acc}, Test loss: {test_loss}')

    # Plotting
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
    data_path = "C:/Users/Pana/Desktop/Northumbria/Final Year/Individual Computing Project " \
                "KV6003BNN01/Speech-Emotion-Recognition---Audio-Dataset/dataset/CREMAD"
    X_train, X_test, y_train, y_test = load_data(data_path)
    model = build_model(X_train.shape[1])
    train_and_evaluate(model, X_train, X_test, y_train, y_test)
