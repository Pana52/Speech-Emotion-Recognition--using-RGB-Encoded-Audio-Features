import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
import os
from PIL import Image

# Path to your dataset of Mel spectrogram images
data_path = 'C:/Users/Pana/Desktop/Northumbria/Final Year/Individual Computing Project ' \
            'KV6003BNN01/Speech-Emotion-Recognition---Audio-Dataset/models/deep learning for ' \
            'images/datasets/CREMAD/MELSPEC_224x224/'


def load_dataset(dataset_path):
    images = []
    labels = []
    class_labels = {'ANG': 0, 'DIS': 1, 'FEA': 2, 'HAP': 3, 'NEU': 4, 'SAD': 5}  # Adjust as needed

    for root, dirs, files in os.walk(dataset_path):
        for file in files:
            if file.endswith('.png'):  # Adjust the file format as needed
                path = os.path.join(root, file)
                img = Image.open(path).convert('L')  # 'L' for grayscale, 'RGB' for color
                img = img.resize((224, 224))  # Adjust the target size as per your model's input
                img_array = np.array(img)
                images.append(img_array)

                # Correctly extract label from the path
                label_dir = os.path.basename(os.path.normpath(root))
                try:
                    label = class_labels[label_dir]
                except KeyError:
                    print(f"Label '{label_dir}' not found in class_labels dictionary.")
                    continue  # Skip this file if its label is not in class_labels

                labels.append(label)

    images = np.array(images)
    labels = np.array(labels)

    images = np.expand_dims(images, axis=-1)  # Add channel dimension for grayscale images
    labels = np.eye(len(class_labels))[labels]  # One-hot encode labels

    return images, labels


def load_data(dataset_path, test_size=0.2, val_size=0.2):
    X, y = load_dataset(dataset_path)
    X = X.astype('float32') / 255.0  # Normalize pixel values
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=test_size + val_size)
    test_ratio = test_size / (test_size + val_size)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=test_ratio)

    return (X_train, y_train), (X_val, y_val), (X_test, y_test)


def create_generators(batch_size=32):
    datagen = ImageDataGenerator(
        rescale=1. / 255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )

    (X_train, y_train), (X_val, y_val), (X_test, y_test) = load_data(data_path)

    train_generator = datagen.flow(X_train, y_train, batch_size=batch_size)
    validation_generator = datagen.flow(X_val, y_val, batch_size=batch_size)
    test_generator = datagen.flow(X_test, y_test, batch_size=batch_size)

    return train_generator, validation_generator, test_generator
