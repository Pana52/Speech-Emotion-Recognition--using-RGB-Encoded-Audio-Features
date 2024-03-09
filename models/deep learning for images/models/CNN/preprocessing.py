import os
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
import pandas as pd  # For label encoding
from skimage.transform import rotate, AffineTransform, warp
from skimage.util import random_noise
from skimage.exposure import adjust_gamma
import random


def load_images_from_folder(folder_path, image_size=(100, 100)):
    images = []
    labels = []
    for label_dir in os.listdir(folder_path):
        class_dir = os.path.join(folder_path, label_dir)
        if not os.path.isdir(class_dir):
            continue
        for file in os.listdir(class_dir):
            if file.endswith(".png"):
                img_path = os.path.join(class_dir, file)
                img = Image.open(img_path).convert('RGB')
                img = img.resize(image_size)
                img = np.array(img)
                images.append(img)
                labels.append(label_dir)
    images = np.array(images, dtype='float32')
    labels = np.array(labels)
    return images, labels


def preprocess_data(X, y):
    X = np.expand_dims(X, -1)  # Add channel dimension
    X = X / 255.0  # Normalize images to [0, 1]
    y = pd.get_dummies(y).values  # One-hot encode labels
    return X, y


def split_dataset(X, y, test_size=0.2, val_size=0.25):
    # First split to get training and initial test set
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
    # Split training set to obtain a validation set
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=val_size, random_state=42)
    return X_train, X_val, X_test, y_train, y_val, y_test


def apply_augmentations(images, augmentations=None):
    if augmentations is None:
        augmentations = {}

    augmented_images = []
    for img in images:
        if augmentations.get('horizontal_flip', False) and random.choice([True, False]):
            img = np.fliplr(img)

        if augmentations.get('vertical_flip', False) and random.choice([True, False]):
            img = np.flipud(img)

        if 'rotation' in augmentations:
            angle = random.uniform(-augmentations['rotation'], augmentations['rotation'])
            img = rotate(img, angle, mode='edge')

        if 'noise' in augmentations:
            img = random_noise(img, mode='gaussian', var=augmentations['noise'] ** 2)

        if 'brightness' in augmentations:
            factor = random.uniform(1 - augmentations['brightness'], 1 + augmentations['brightness'])
            img = adjust_gamma(img, gamma=factor, gain=1)

        if 'shear' in augmentations:
            af_transform = AffineTransform(shear=np.deg2rad(augmentations['shear']))
            img = warp(img, inverse_map=af_transform)

        augmented_images.append(img)

    return np.array(augmented_images)
