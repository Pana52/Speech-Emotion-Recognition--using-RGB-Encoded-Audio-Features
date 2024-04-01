import numpy as np
import os
from keras_preprocessing.image import load_img, img_to_array
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import to_categorical
from skimage.transform import resize


def load_and_preprocess_images(directory_path, target_size=(224, 224)):
    """
    Loads images from the specified directory and preprocesses them for CNN input.

    Parameters:
    - directory_path: str, the path to the directory containing the images.
    - target_size: tuple of (height, width), the target size of the images.

    Returns:
    - images: np.array, array of preprocessed images.
    - labels: np.array, array of labels corresponding to each image.
    """
    images = []
    labels = []
    label_map = {}

    for idx, folder_name in enumerate(os.listdir(directory_path)):
        folder_path = os.path.join(directory_path, folder_name)
        if os.path.isdir(folder_path):
            label_map[folder_name] = idx
            for file_name in os.listdir(folder_path):
                if file_name.endswith('.png'):
                    file_path = os.path.join(folder_path, file_name)
                    image = load_img(file_path, target_size=target_size, color_mode='rgb')
                    image = img_to_array(image) / 255.0  # Normalize to [0, 1]
                    images.append(image)
                    labels.append(idx)

    images = np.array(images)
    labels = to_categorical(np.array(labels), num_classes=len(label_map))

    return images, labels


def augment_images(images):
    """
    Creates a data generator for augmenting images.

    Parameters:
    - images: np.array, array of images to augment.

    Returns:
    - datagen: An ImageDataGenerator instance with augmentation defined.
    """
    datagen = ImageDataGenerator(
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )

    datagen.fit(images)
    return datagen
