"""
cnn_feature_extractor.py
------------------------

This module defines a Convolutional Neural Network (CNN) for extracting features from Mel-Spectrogram images.

Functions:
- build_cnn_model: Constructs and compiles the CNN model.
- extract_features: Uses the trained CNN model to extract features from Mel-Spectrogram images.
"""
import numpy as np
from keras.models import Sequential, Model
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense


def build_cnn_model(input_shape=(224, 224, 3), num_classes=10):
    """
    Builds a simple CNN model for feature extraction.

    Parameters:
    - input_shape: tuple, the shape of the input images.
    - num_classes: int, the number of classes.

    Returns:
    - model: a compiled CNN model.
    """
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(128, activation='relu'),
        Dense(num_classes, activation='softmax')
    ])

    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    return model


def extract_features(model, images):
    """
    Extracts features from images using the specified CNN model.

    Parameters:
    - model: Sequential, the CNN model for feature extraction.
    - images: np.array, a batch of preprocessed images.

    Returns:
    - features: np.array, extracted features from the CNN model.
    """
    # Assuming the feature extraction layer is the penultimate layer of the model
    feature_extractor = Model(inputs=model.inputs, outputs=model.layers[-3].output)

    features = feature_extractor.predict(images)
    return features
