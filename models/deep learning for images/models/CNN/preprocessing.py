from keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
import os
import numpy as np
from glob import glob
import pandas as pd


def split_data(dataset_path, validation_split=0.2, test_split=0.1):
    """
    Splits the dataset into training, validation, and testing paths.
    """
    all_files = glob(os.path.join(dataset_path, '*/*'))
    all_classes = [os.path.basename(os.path.dirname(f)) for f in all_files]

    # Splitting dataset into train and test initially
    train_files, test_files, train_classes, test_classes = train_test_split(
        all_files, all_classes, test_size=test_split, stratify=all_classes)

    # Splitting the training dataset further into train and validation sets
    train_files, validation_files, train_classes, validation_classes = train_test_split(
        train_files, train_classes, test_size=validation_split, stratify=train_classes)

    return train_files, validation_files, test_files


def create_data_generators(dataset_path, target_size, batch_size, validation_split=0.2, test_split=0.1):
    """
    Creates data generators for training, validation, and test datasets.
    """
    # Getting the full paths for all images
    all_image_files = glob(os.path.join(dataset_path, '*/*.png'))
    np.random.shuffle(all_image_files)  # Shuffling the files is important before splitting

    # Extracting class labels from the file paths
    all_classes = [os.path.basename(os.path.dirname(f)) for f in all_image_files]
    all_data = pd.DataFrame({
        'filename': all_image_files,
        'class': all_classes
    })

    # Split the dataset into training+validation and test sets
    train_val_data, test_data = train_test_split(all_data, test_size=test_split, stratify=all_data['class'],
                                                 shuffle=True)

    # Split the training+validation set into training and validation sets
    train_data, validation_data = train_test_split(train_val_data, test_size=validation_split / (1 - test_split),
                                                   stratify=train_val_data['class'], shuffle=True)

    # Instantiate the ImageDataGenerator class (with augmentation for training set only)
    train_datagen = ImageDataGenerator(
        rescale=1. / 255,
        # rotation_range=20,
        # width_shift_range=0.2,
        # height_shift_range=0.2,
        # shear_range=0.2,
        # zoom_range=0.2,
        # horizontal_flip=True,
        fill_mode='nearest'
    )

    validation_datagen = ImageDataGenerator(rescale=1. / 255)
    test_datagen = ImageDataGenerator(rescale=1. / 255)

    # Create the actual generators
    train_generator = train_datagen.flow_from_dataframe(
        dataframe=train_data,
        directory=None,
        x_col='filename',
        y_col='class',
        target_size=target_size,
        batch_size=batch_size,
        class_mode='categorical'
    )

    validation_generator = validation_datagen.flow_from_dataframe(
        dataframe=validation_data,
        directory=None,
        x_col='filename',
        y_col='class',
        target_size=target_size,
        batch_size=batch_size,
        class_mode='categorical'
    )

    test_generator = test_datagen.flow_from_dataframe(
        dataframe=test_data,
        directory=None,
        x_col='filename',
        y_col='class',
        target_size=target_size,
        batch_size=batch_size,
        class_mode='categorical'
    )

    # Return the data generators
    return train_generator, validation_generator, test_generator
