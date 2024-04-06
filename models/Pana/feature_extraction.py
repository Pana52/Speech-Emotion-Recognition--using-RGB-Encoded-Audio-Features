import os
import librosa
import numpy as np
import matplotlib.pyplot as plt
from keras.utils import load_img, img_to_array
from librosa.display import specshow
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from keras.preprocessing import image
from keras.applications.resnet import ResNet50, preprocess_input

# Configuration
DATASET_PATH = "C:/Users/Pana/Desktop/Northumbria/Final Year/Individual Computing Project " \
               "KV6003BNN01/datasets/Mixed/EMODB/"
EMOTIONS = ['anger', 'boredom', 'disgust', 'fear', 'happiness', 'neutral', 'sadness']
DURATION = 2.5
SAMPLE_RATE = 48000
SAMPLES = int(SAMPLE_RATE * DURATION)
N_CLUSTERS = 10
N_MELS = 128
HOP_L = 1024
N_FFT = 2048


def load_and_preprocess_audio(file_path, target_sr=SAMPLE_RATE, target_length=SAMPLES):
    audio, sr = librosa.load(file_path, sr=target_sr)
    if len(audio) < target_length:
        pad_length = target_length - len(audio)
        audio = np.pad(audio, (0, pad_length), mode='constant')
    else:
        audio = audio[:target_length]
    return audio


def extract_mfcc(audio, sr=SAMPLE_RATE, n_mfcc=13):
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=n_mfcc)
    mfcc_scaled = np.mean(mfcc.T, axis=0)
    return mfcc_scaled


def create_mel_spectrogram(audio, sr=SAMPLE_RATE, save_path='mel_spectrogram.png', n_mels=N_MELS, hop_length=HOP_L,
                           n_fft=N_FFT):
    S = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=n_mels, hop_length=hop_length, n_fft=n_fft)
    S_DB = librosa.power_to_db(S, ref=np.max)

    # Determine figure size for 256x256 image resolution
    fig, ax = plt.subplots(figsize=(2.56, 2.56), dpi=100)
    plt.axis('off')  # Ensure no axes are shown

    # Displaying the mel spectrogram without axis labels
    specshow(S_DB, sr=sr, hop_length=hop_length)
    plt.tight_layout(pad=0)

    # Save the spectrogram as an image file, ensuring no whitespace or padding
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0, dpi=100)
    plt.close(fig)


def apply_clustering(features, n_clusters=N_CLUSTERS):
    """
    Applies KMeans clustering to the features and transforms them based on the cluster centroids.
    """
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    # Fit and transform the features to the cluster-distance space
    clustered_features = kmeans.fit_transform(features)
    return clustered_features


model = ResNet50(weights='imagenet', include_top=False, input_shape=(256, 256, 3))


def extract_image_features(image_path):
    # Load and preprocess the image
    img = load_img(image_path, target_size=(256, 256))  # Resize image to match model's expected input dimensions
    img_array = img_to_array(img)  # Convert the PIL Image to a numpy array
    img_array_expanded_dims = np.expand_dims(img_array, axis=0)  # Add an extra dimension for the batch size
    preprocessed_img = preprocess_input(img_array_expanded_dims)  # Preprocess the image using ResNet50's preprocessing

    # Predict the features of the image using the model
    features = model.predict(preprocessed_img)

    # Flatten the features to a 1D vector to concatenate with audio features
    flattened_features = features.flatten()

    return flattened_features


def process_dataset(dataset_path=DATASET_PATH, emotions=EMOTIONS, test_size=0.2, validation_size=0.2):
    combined_features, labels = [], []  # Lists to store combined audio+image features and labels

    audio_base_path = os.path.join(dataset_path, "Audio")
    image_base_path = os.path.join(dataset_path, "Images")

    for emotion in emotions:
        audio_emotion_path = os.path.join(audio_base_path, emotion)
        image_emotion_path = os.path.join(image_base_path, emotion)

        for filename in os.listdir(audio_emotion_path):
            if not filename.endswith(".wav"):  # Process only .wav files
                continue

            audio_file_path = os.path.join(audio_emotion_path, filename)
            image_file_path = os.path.join(image_emotion_path,
                                           filename.replace(".wav", ".png"))  # Assuming image format is .png

            audio = load_and_preprocess_audio(audio_file_path)
            mfcc = extract_mfcc(audio)

            image_features = extract_image_features(image_file_path)  # Extract image features

            combined_feature = np.concatenate((mfcc, image_features))  # Combine audio and image features
            combined_features.append(combined_feature)
            labels.append(emotion)

    # Convert lists to NumPy arrays for clustering
    combined_features = np.array(combined_features)

    # Apply clustering to the MFCC features
    clustered_features = apply_clustering(combined_features, n_clusters=N_CLUSTERS)

    # Splitting the dataset into training, validation, and testing sets
    X_train, X_test, y_train, y_test = train_test_split(clustered_features, labels, test_size=test_size,
                                                        stratify=labels, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=validation_size / (1 - test_size),
                                                      stratify=y_train, random_state=42)

    # Save the datasets
    save_dataset(X_train, y_train, 'train')
    save_dataset(X_test, y_test, 'test')
    save_dataset(X_val, y_val, 'validation')


def save_dataset(features, labels, dataset_type):
    np.savez_compressed(f'{dataset_type}_dataset.npz', features=features, labels=labels)


if __name__ == '__main__':
    process_dataset()
