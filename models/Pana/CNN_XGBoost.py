import numpy as np
import librosa
import os
from keras.applications.resnet import ResNet50, preprocess_input
from keras.models import Model
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import load_img, img_to_array
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


# Constants
IMAGE_SIZE = (224, 224)
SAMPLE_RATE = 48000

# Initialize ResNet50 for feature extraction
base_model = ResNet50(weights='imagenet')
model = Model(inputs=base_model.input, outputs=base_model.get_layer('avg_pool').output)


# Audio Augmentation Functions
def add_noise(audio, noise_level=0.005):
    noise = np.random.randn(len(audio))
    augmented_audio = audio + noise_level * noise
    augmented_audio = np.clip(augmented_audio, -1.0, 1.0)
    return augmented_audio


def time_shift(audio, shift_max=0.2):
    shift = int(np.random.random() * shift_max * len(audio))
    return np.roll(audio, shift)


def change_speed(audio, speed_factor=1.25):
    return librosa.effects.time_stretch(audio, rate=speed_factor)


def change_pitch(audio, sampling_rate, pitch_factor=5):
    return librosa.effects.pitch_shift(audio, sampling_rate, pitch_factor)


# Image Augmentation Setup
def create_image_datagen():
    return ImageDataGenerator(
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest')


# Loading and Preprocessing Image
def load_and_preprocess_image(img_path):
    img = load_img(img_path, target_size=IMAGE_SIZE)  # Adjusted to new IMAGE_SIZE
    img = img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = preprocess_input(img)
    return img


# Main function to process and augment data, then train and evaluate the model

def main():
    # Example paths - replace these with paths to your actual data
    audio_path = "C:/Users/Pana/Desktop/Northumbria/Final Year/Individual Computing Project " \
                 "KV6003BNN01/datasets/Mixed/EMODB/audio/"
    image_path = "C:/Users/Pana/Desktop/Northumbria/Final Year/Individual Computing Project " \
                 "KV6003BNN01/datasets/Mixed/EMODB/images/"

    features = []
    labels = []

    # Create a mapping of emotion categories to numeric labels
    # Adjust the mapping as per your specific emotion categories
    emotion_labels = {
        'anger': 0,
        'boredom': 1,
        'disgust': 2,
        'fear': 3,
        'happiness': 4,
        'neutral': 5,
        'sadness': 6,
    }

    # Process each emotion category under audio
    for emotion_dir in os.listdir(audio_path):
        emotion_audio_path = os.path.join(audio_path, emotion_dir)
        for audio_file in os.listdir(emotion_audio_path):
            full_audio_path = os.path.join(emotion_audio_path, audio_file)
            audio, sr = librosa.load(full_audio_path, sr=SAMPLE_RATE)
            augmented_audio = add_noise(audio)
            augmented_audio = time_shift(augmented_audio)
            augmented_audio = change_speed(augmented_audio)
            # Additional audio processing here
            # Assuming features extracted from audio would also be used for classification
            # You'd need to integrate audio feature extraction here

    # Process each emotion category under images
    for emotion_dir in os.listdir(image_path):
        emotion_image_path = os.path.join(image_path, emotion_dir)
        for image_file in os.listdir(emotion_image_path):
            full_image_path = os.path.join(emotion_image_path, image_file)
            img = load_and_preprocess_image(full_image_path)
            datagen = create_image_datagen()
            it = datagen.flow(img, batch_size=1)
            batch = next(it)
            augmented_img = batch[0]
            augmented_img = np.expand_dims(augmented_img, axis=0)

            # Extract features
            extracted_features = model.predict(augmented_img)
            features.append(extracted_features.flatten())

            # Assign labels based on the emotion directory
            if emotion_dir in emotion_labels:
                labels.append(emotion_labels[emotion_dir])
            else:
                print(f"Warning: Unrecognized emotion category '{emotion_dir}'")

    # Convert features and labels to numpy arrays
    features = np.array(features)
    labels = np.array(labels)

    # Splitting and model training
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)
    gbm = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)

    print(np.unique(labels))  # Should now show more than one unique label

    gbm.fit(X_train, y_train)
    predictions = gbm.predict(X_test)
    print(f"GBM Classification Accuracy: {accuracy_score(y_test, predictions)}")
    print(classification_report(y_test, predictions))
    print(confusion_matrix(y_test, predictions))


if __name__ == "__main__":
    main()

