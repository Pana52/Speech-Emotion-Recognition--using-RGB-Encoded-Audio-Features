import os
import numpy as np
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array


def create_data_generators(dataset_path, target_size=(32, 32), batch_size=32, validation_split=0.2, test_split=0.1):
    # Enhanced ImageDataGenerator for training with additional data augmentation techniques
    train_datagen = ImageDataGenerator(
        rescale=1. / 255,
        fill_mode='nearest',  # Strategy used for filling in newly created pixels
        validation_split=validation_split + test_split  # Combine validation and test for splitting

    )

    # Define ImageDataGenerator for validation and test without data augmentation
    test_val_datagen = ImageDataGenerator(
        rescale=1. / 255,
        validation_split=validation_split + test_split
    )

    # Setup train, validation, and test generators
    train_generator = train_datagen.flow_from_directory(
        dataset_path,
        target_size=target_size,
        batch_size=batch_size,
        class_mode='categorical',
        subset='training',  # Set as training data
        seed=42
    )

    validation_generator = test_val_datagen.flow_from_directory(
        dataset_path,
        target_size=target_size,
        batch_size=batch_size,
        class_mode='categorical',
        subset='validation',  # Set as validation data
        seed=42
    )

    test_generator = test_val_datagen.flow_from_directory(
        dataset_path,
        target_size=target_size,
        batch_size=batch_size,
        class_mode='categorical',
        subset='validation',  # Intentionally using 'validation' to split data
        seed=42
    )

    return train_generator, validation_generator, test_generator


def load_dataset(dataset_path, image_size=(32, 32)):

    images = []
    labels = []

    # Assuming the dataset_path contains subdirectories named after the labels
    for label in os.listdir(dataset_path):
        label_path = os.path.join(dataset_path, label)
        if os.path.isdir(label_path):
            for image_file in os.listdir(label_path):
                image_path = os.path.join(label_path, image_file)
                image = load_img(image_path, target_size=image_size)
                image = img_to_array(image)
                images.append(image)
                labels.append(label)

    X = np.array(images, dtype="float32") / 255.0  # Normalize to [0, 1]
    y = np.array(labels)

    return X, y


def split_dataset(X, y, test_size=0.2, val_size=0.2):

    # Splitting dataset into training and test set
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, stratify=y, random_state=42)

    # Further split training set to create a validation set
    val_size_adjusted = val_size / (
                1 - test_size)  # Adjust validation size based on the remaining dataset after test split
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=val_size_adjusted, stratify=y_train,
                                                      random_state=42)

    return X_train, X_val, X_test, y_train, y_val, y_test
