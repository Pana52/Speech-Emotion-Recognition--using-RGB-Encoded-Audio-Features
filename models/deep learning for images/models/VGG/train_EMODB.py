from keras.models import Model
from keras.layers import Flatten, Dense, Input, GlobalAveragePooling2D
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from PIL import Image
import numpy as np
import os
from keras.callbacks import EarlyStopping
from keras.applications import VGG16
from keras.applications.vgg16 import preprocess_input

IMAGE_SIZE = (32, 32)
INPUT_SHAPE = (32, 32, 3)
BATCH_SIZE = 32
NUM_CLASSES = 7
PATIENCE = 50
EPOCHS = 1000


# Adjusted load_and_preprocess_data function
def load_and_preprocess_data(dataset_path):
    classes = os.listdir(dataset_path)
    class_labels = {class_name: index for index, class_name in enumerate(classes)}
    X = []
    y = []

    for class_name, class_index in class_labels.items():
        class_path = os.path.join(dataset_path, class_name)
        for image_file in os.listdir(class_path):
            image_path = os.path.join(class_path, image_file)
            image = Image.open(image_path).convert('RGB')
            image = image.resize(IMAGE_SIZE)
            image = np.array(image)
            image = preprocess_input(image)  # Use VGG16's preprocess_input
            X.append(image)
            y.append(class_index)
    X = np.array(X)
    y = to_categorical(y, num_classes=len(classes))

    return train_test_split(X, y, test_size=0.2, random_state=42), class_labels


def create_model(input_shape, num_classes):
    base_model = VGG16(include_top=False, weights='imagenet', input_tensor=Input(shape=input_shape))
    base_model.trainable = False  # Freeze the base model initially

    x = base_model.output
    x = GlobalAveragePooling2D()(x)  # Use GlobalAveragePooling for dimensionality reduction
    x = Dense(128, activation='relu')(x)
    x = Dense(128, activation='relu')(x)
    predictions = Dense(num_classes, activation='softmax')(x)

    model = Model(inputs=base_model.input, outputs=predictions)
    return model


# Function to compile and train the model, including the classification report
def compile_and_train_model(model, X_train, y_train, X_val, y_val, class_labels, epochs=EPOCHS, batch_size=BATCH_SIZE):
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    early_stopping = EarlyStopping(monitor='val_loss', min_delta=0.001, patience=PATIENCE, verbose=1,
                                   restore_best_weights=True)

    history = model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs, validation_data=(X_val, y_val),
                        callbacks=early_stopping)

    # Predict classes on the validation set
    y_pred = model.predict(X_val, batch_size=batch_size)
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_true_classes = np.argmax(y_val, axis=1)

    # Generate and print the classification report
    report = classification_report(y_true_classes, y_pred_classes, target_names=list(class_labels.keys()))
    print(report)

    return history


if __name__ == "__main__":
    dataset_path = 'C:/Users/Pana/Desktop/Northumbria/Final Year/Individual Computing Project ' \
                   'KV6003BNN01/Speech-Emotion-Recognition---Audio-Dataset/models/deep learning for ' \
                   'images/datasets/EMODB/MFCCs/MFCC_32x32/'
    (X_train, X_val, y_train, y_val), class_labels = load_and_preprocess_data(dataset_path)
    model = create_model(input_shape=INPUT_SHAPE, num_classes=NUM_CLASSES)
    history = compile_and_train_model(model, X_train, y_train, X_val, y_val, class_labels)
