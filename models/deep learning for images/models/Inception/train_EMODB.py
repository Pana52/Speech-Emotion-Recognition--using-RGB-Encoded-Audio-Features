import os
import numpy as np
from keras_preprocessing.image import img_to_array, load_img
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from keras.layers import Dense, GlobalAveragePooling2D, Input
from keras.models import Model
from keras.optimizers import Adam
from keras.callbacks import Callback
from keras.applications.inception_v3 import InceptionV3, preprocess_input

# Constants
EPOCH = 1000
PATIENCE = 50
SWITCH_EPOCH = 20
ORIGINAL_IMAGE_SIZE = (256, 256)
TARGET_IMAGE_SIZE = (256, 256)
BATCH_SIZE = 32
NUM_CLASSES = 7


# Custom Early Stopping
class CustomEarlyStopping(Callback):
    def __init__(self, switch_epoch, min_delta=0, patience=0):
        super().__init__()
        self.switch_epoch = switch_epoch
        self.min_delta = min_delta
        self.patience = patience
        self.best_weights = None

        self.best_val_loss = np.Inf
        self.wait_loss = 0

        self.best_val_acc = 0
        self.wait_acc = 0

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


# Data Preprocessing
# Function to load and preprocess data
# Adjusted function to load and preprocess data with resizing
def load_and_preprocess_data(dataset_path, target_size=TARGET_IMAGE_SIZE):
    X, y = [], []
    classes = os.listdir(dataset_path)
    classes.sort()  # Ensure consistent order
    class_indices = {class_name: index for index, class_name in enumerate(classes)}
    for class_name, index in class_indices.items():
        class_path = os.path.join(dataset_path, class_name)
        for img_name in os.listdir(class_path):
            img_path = os.path.join(class_path, img_name)
            img = load_img(img_path, target_size=target_size)
            img = img_to_array(img)
            img = preprocess_input(img)
            X.append(img)
            y.append(index)
    X = np.array(X)
    y = np.array(y)
    return train_test_split(X, y, test_size=0.2, random_state=42), list(class_indices.keys())



# Building the model remains the same
def build_model(num_classes=NUM_CLASSES):
    input_tensor = Input(shape=(TARGET_IMAGE_SIZE[0], TARGET_IMAGE_SIZE[1], 3))
    base_model = InceptionV3(include_top=False, weights='imagenet', input_tensor=input_tensor)
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    predictions = Dense(num_classes, activation='softmax')(x)
    model = Model(inputs=base_model.input, outputs=predictions)
    for layer in base_model.layers:
        layer.trainable = False
    model.compile(optimizer=Adam(learning_rate=0.001), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model


if __name__ == "__main__":
    dataset_path = 'C:/Users/Pana/Desktop/Northumbria/Final Year/Individual Computing Project ' \
                   'KV6003BNN01/Speech-Emotion-Recognition---Audio-Dataset/models/deep learning for ' \
                   'images/datasets/EMODB/MFCCs/MFCC_256x256/'
    (X_train, X_test, y_train, y_test), target_names = load_and_preprocess_data(dataset_path)
    model = build_model(NUM_CLASSES)
    custom_early_stopping = CustomEarlyStopping(switch_epoch=SWITCH_EPOCH, min_delta=0.001, patience=PATIENCE)

    # Fit the model
    history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=EPOCH, batch_size=BATCH_SIZE,
                        callbacks=[custom_early_stopping])

    # Make predictions
    y_pred = model.predict(X_test, batch_size=BATCH_SIZE)
    # Convert predictions from one-hot encoded to class numbers
    y_pred_classes = np.argmax(y_pred, axis=1)

    # Generate classification report
    print(classification_report(y_test, y_pred_classes, target_names=target_names))