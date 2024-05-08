# Import necessary libraries
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from keras.callbacks import Callback
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from PIL import Image
import numpy as np
import os

EPOCH = 100
PATIENCE = 20
SWITCH_EPOCH = 20
IMAGE_SIZE = (256, 256)
BATCH_SIZE = 32


# CustomEarlyStopping Class
class CustomEarlyStopping(Callback):
    def __init__(self, switch_epoch, min_delta=0, patience=0):
        super().__init__()
        self.switch_epoch = switch_epoch  # Epoch to switch from val_loss to val_accuracy
        self.min_delta = min_delta
        self.patience = patience
        self.best_weights = None  # Store the best weights

        self.best_val_loss = np.Inf
        self.wait_loss = 0  # Counter for patience mechanism for loss

        self.best_val_acc = 0
        self.wait_acc = 0  # Counter for patience mechanism for accuracy

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        val_loss = logs.get('val_loss')
        val_acc = logs.get('val_accuracy')

        if epoch < self.switch_epoch:
            if np.less(val_loss - self.min_delta, self.best_val_loss):
                self.best_val_loss = val_loss
                self.wait_loss = 0
                self.best_weights = self.model.get_weights()
            else:
                self.wait_loss += 1
                if self.wait_loss >= self.patience:
                    print(f"\nEpoch {epoch + 1}: early stopping (minimizing val_loss)")
                    self.model.stop_training = True
                    self.model.set_weights(self.best_weights)
        else:
            if np.greater(val_acc - self.min_delta, self.best_val_acc):
                self.best_val_acc = val_acc
                self.wait_acc = 0
                self.best_weights = self.model.get_weights()
            else:
                self.wait_acc += 1
                if self.wait_acc >= self.patience:
                    print(f"\nEpoch {epoch + 1}: early stopping (maximizing val_accuracy)")
                    self.model.stop_training = True
                    self.model.set_weights(self.best_weights)


# Function to load and preprocess data
def load_and_preprocess_data(dataset_path):
    classes = os.listdir(dataset_path)
    class_labels = {class_name: index for index, class_name in enumerate(classes)}
    X = []  # Image data
    y = []  # Labels

    for class_name, class_index in class_labels.items():
        class_path = os.path.join(dataset_path, class_name)
        for image_file in os.listdir(class_path):
            image_path = os.path.join(class_path, image_file)
            image = Image.open(image_path).convert('RGB')  # Ensure 3 channels
            image = image.resize(IMAGE_SIZE)  # Resize to a common size
            X.append(np.array(image))
            y.append(class_index)
    X = np.array(X) / 255.0  # Normalize pixel values
    y = to_categorical(y, num_classes=len(classes))  # Convert labels to one-hot encoding

    # Split into training and validation sets
    return train_test_split(X, y, test_size=0.2, random_state=42), class_labels


# Function to create the 3CF_CNN model
def create_model(input_shape, num_classes):
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ])
    return model


# Function to compile and train the model, including the classification report
def compile_and_train_model(model, X_train, y_train, X_val, y_val, class_labels, epochs=EPOCH, batch_size=BATCH_SIZE):  # Added batch_size parameter
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    custom_early_stopping = CustomEarlyStopping(switch_epoch=SWITCH_EPOCH, min_delta=0.001, patience=PATIENCE)
    history = model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs, validation_data=(X_val, y_val), callbacks=[custom_early_stopping])  # Added batch_size here

    # Predict classes on the validation set
    y_pred = model.predict(X_val, batch_size=batch_size)  # Added batch_size here for consistency in prediction
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_true_classes = np.argmax(y_val, axis=1)

    # Generate and print the classification report
    report = classification_report(y_true_classes, y_pred_classes, target_names=list(class_labels.keys()))
    print(report)

    return history


if __name__ == "__main__":
    dataset_path = 'PATH'
    (X_train, X_val, y_train, y_val), class_labels = load_and_preprocess_data(dataset_path)
    model = create_model(input_shape=X_train.shape[1:], num_classes=y_train.shape[1])
    history = compile_and_train_model(model, X_train, y_train, X_val, y_val, class_labels)
