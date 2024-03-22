import os
from keras.preprocessing.image import ImageDataGenerator


def load_data(dataset_path, img_size=(128, 128), val_size=0.2):
    # Define data generators with augmentation for the training data
    train_datagen = ImageDataGenerator(
        rescale=1. / 255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest',
        validation_split=val_size  # Specify the validation split
    )

    validation_datagen = ImageDataGenerator(
        rescale=1. / 255,
        validation_split=val_size  # Specify the validation split
    )

    train_generator = train_datagen.flow_from_directory(
        dataset_path,
        target_size=img_size,
        batch_size=32,
        class_mode='categorical',
        subset='training',  # Set as training data
        color_mode='grayscale'
    )

    validation_generator = validation_datagen.flow_from_directory(
        dataset_path,
        target_size=img_size,
        batch_size=32,
        class_mode='categorical',
        subset='validation',  # Set as validation data
        color_mode='grayscale'
    )

    return {
        'train': train_generator,
        'validation': validation_generator
    }
