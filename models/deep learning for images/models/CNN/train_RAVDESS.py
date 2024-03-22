import numpy as np
import os
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint, EarlyStopping
from sklearn.metrics import classification_report
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization, Activation
from CustomEarlyStopping import CustomEarlyStopping

# CONSTANTS AND VAARIABLES
DATA_PATH = 'C:/Users/Pana/Desktop/Northumbria/Final Year/Individual Computing Project ' \
            'KV6003BNN01/Speech-Emotion-Recognition---Audio-Dataset/models/deep learning for ' \
            'images/datasets/RAVDESS/Mel-Spectrograms/MelSpec_32x32/'
NUM_CLASSES = 8
INPUT_SHAPE = (32, 32, 3)
IMG_SIZE = (32, 32)
EPOCHS = 1000
BATCH_SIZE = 32

# CUSTOM EARLY STOPPING CONSTANTS
MONITOR = "val_loss"
PATIENCE = 20
MODE = "min"
START_EPOCH = 5
SWITCH_EPOCH = 50 + START_EPOCH


def load_data(dataset_path, img_size=IMG_SIZE, val_size=0.2):
    """
    Load and preprocess the dataset.
    """
    train_datagen = ImageDataGenerator(
        rescale=1. / 255,
        # rotation_range=20,
        # width_shift_range=0.2,
        # height_shift_range=0.2,
        # shear_range=0.2,
        # zoom_range=0.2,
        # horizontal_flip=True,
        fill_mode='nearest',
        validation_split=val_size
    )

    validation_datagen = ImageDataGenerator(
        rescale=1. / 255,
        validation_split=val_size
    )

    train_generator = train_datagen.flow_from_directory(
        dataset_path,
        target_size=img_size,
        batch_size=32,
        class_mode='categorical',
        subset='training',
        color_mode='rgb'
    )

    validation_generator = validation_datagen.flow_from_directory(
        dataset_path,
        target_size=img_size,
        batch_size=32,
        class_mode='categorical',
        subset='validation',
        color_mode='rgb'
    )

    return train_generator, validation_generator


def build_model(input_shape=INPUT_SHAPE, num_classes=NUM_CLASSES):
    """
    Build and compile the CNN model with batch normalization.
    """
    model = Sequential([
        # First Conv Block
        Conv2D(32, (3, 3), padding='same', input_shape=input_shape),
        BatchNormalization(),
        Activation('relu'),
        MaxPooling2D((2, 2)),

        # Second Conv Block
        Conv2D(64, (3, 3), padding='same'),
        BatchNormalization(),
        Activation('relu'),
        MaxPooling2D((2, 2)),

        # Third Conv Block
        Conv2D(128, (3, 3), padding='same'),
        BatchNormalization(),
        Activation('relu'),
        MaxPooling2D((2, 2)),

        # Fully Connected Layers
        Flatten(),
        Dense(128),
        BatchNormalization(),
        Activation('relu'),
        Dropout(0.5),

        # Output Layer
        Dense(num_classes, activation='softmax')
    ])

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model


def train_model(dataset_path, input_shape=INPUT_SHAPE, num_classes=NUM_CLASSES, epochs=EPOCHS, batch_size=BATCH_SIZE):
    train_generator, validation_generator = load_data(dataset_path, img_size=input_shape[:2])

    model = build_model(input_shape, num_classes)

    # callbacks = EarlyStopping(monitor='val_accuracy', patience=PATIENCE, verbose=1, mode='max')

    custom_early_stopping = CustomEarlyStopping(switch_epoch=SWITCH_EPOCH, min_delta=0.001, patience=PATIENCE)

    model_checkpoint = ModelCheckpoint(
        'best_model.h5',
        save_best_only=True,
        monitor='val_loss',
        mode='min'
    )

    history = model.fit(
        train_generator,
        steps_per_epoch=train_generator.samples // batch_size,
        epochs=epochs,
        validation_data=validation_generator,
        validation_steps=validation_generator.samples // batch_size,
        callbacks=custom_early_stopping
        # callbacks=callbacks
    )

    validation_generator.reset()
    Y_pred = model.predict(validation_generator, steps=validation_generator.samples // batch_size + 1)
    y_pred = np.argmax(Y_pred, axis=1)

    print('Classification Report')
    print(classification_report(validation_generator.classes, y_pred,
                                target_names=list(validation_generator.class_indices.keys())))


if __name__ == "__main__":
    dataset_path = DATA_PATH
    train_model(dataset_path)
    print("Model training complete.")
