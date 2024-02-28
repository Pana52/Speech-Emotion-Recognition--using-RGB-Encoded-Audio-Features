from preprocessing_EMODB import load_data

from model import create_model
import numpy as np
from keras.utils import to_categorical
from keras.callbacks import EarlyStopping
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, recall_score, precision_score, \
    f1_score
import matplotlib.pyplot as plt
import seaborn as sns
from keras.models import load_model


def main():
    # Define the path to your dataset
    data_path = "C:/Users/Pana/Desktop/Northumbria/Final Year/Individual Computing Project KV6003BNN01/datasets/EMODB/"

    # Load and preprocess the data
    X_train, X_test, y_train, y_test = load_data(data_path)

    # One-hot encode labels
    num_classes = len(np.unique(y_train))
    y_train_encoded = to_categorical(y_train, num_classes)
    y_test_encoded = to_categorical(y_test, num_classes)

    # Determine input shape
    input_shape = X_train.shape[1:]

    # Create the MLP model
    model = create_model(input_shape, num_classes)

    # Setup Early Stopping
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, verbose=1, mode='min', restore_best_weights=True)

    # Train the model with Early Stopping
    history = model.fit(X_train, y_train_encoded, epochs=100, batch_size=32, validation_data=(X_test, y_test_encoded),
                        callbacks=[early_stopping])

    # Evaluate the model on the test data
    test_loss, test_acc = model.evaluate(X_test, y_test_encoded, verbose=0)
    print(f'Test accuracy: {test_acc}, Test loss: {test_loss}')

    # Make predictions on the test set
    y_pred = model.predict(X_test)
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_true = np.argmax(y_test_encoded, axis=1)

    # Calculate additional metrics
    accuracy = accuracy_score(y_true, y_pred_classes)
    recall = recall_score(y_true, y_pred_classes, average='macro')
    precision = precision_score(y_true, y_pred_classes, average='macro')
    f1 = f1_score(y_true, y_pred_classes, average='macro')

    # Print the computed metrics
    print(f'Accuracy: {accuracy}\nRecall: {recall}\nPrecision: {precision}\nF1-Score: {f1}')

    # Plot training history
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.tight_layout()
    plt.show()

    # Confusion Matrix and Classification Report
    cm = confusion_matrix(y_true, y_pred_classes)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted labels')
    plt.ylabel('True labels')
    plt.title('Confusion Matrix')
    plt.show()

    print('Classification Report:')
    print(classification_report(y_true, y_pred_classes))


if __name__ == '__main__':
    main()
