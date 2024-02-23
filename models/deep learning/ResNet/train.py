# train.py
import numpy as np
from keras.callbacks import ModelCheckpoint, EarlyStopping
from matplotlib import pyplot as plt
from sklearn.metrics import f1_score, recall_score, precision_score, accuracy_score, confusion_matrix, \
    classification_report
from preprocessing import load_data
from model import build_resnet
import seaborn as sns  # Corrected import statement for seaborn

# Constants
DATA_PATH = "C:/Users/Pana/Desktop/Northumbria/Final Year/Individual Computing Project KV6003BNN01/datasets/CREMAD/"
INPUT_SHAPE = (13, 1)  # Adjust this to match the feature dimension of your data
NUM_CLASSES = 6  # Number of emotion categories
EPOCHS = 100
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

    # Make predictions
    y_pred = model.predict(X_test)
    y_pred_classes = np.argmax(y_pred, axis=1)

    # Calculate metrics
    print("Evaluation Metrics:")
    print(f'Accuracy: {accuracy_score(y_test, y_pred_classes)}')
    print(f'Precision: {precision_score(y_test, y_pred_classes, average="macro")}')
    print(f'Recall: {recall_score(y_test, y_pred_classes, average="macro")}')
    print(f'F1-Score: {f1_score(y_test, y_pred_classes, average="macro")}')

    # Plot training history
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.tight_layout()
    plt.show()

    # After evaluating the model
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=2)
    print(f'Test accuracy: {test_acc}, Test loss: {test_loss}')

    # Make predictions
    y_pred_prob = model.predict(X_test)
    y_pred_classes = np.argmax(y_pred_prob, axis=1)

    # Now generate the classification report
    print('Classification Report')
    print(classification_report(y_test, y_pred_classes))

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred_classes)
    plt.figure(figsize=(8, 8))
    sns.heatmap(cm, annot=True, fmt="d", cmap='Blues')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix')
    plt.show()


if __name__ == '__main__':
    main()
