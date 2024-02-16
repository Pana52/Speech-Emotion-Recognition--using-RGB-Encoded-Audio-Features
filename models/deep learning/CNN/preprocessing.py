import os
import librosa
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from keras.utils import to_categorical


# Augmentation function as described earlier
def augment_audio(y, sr, augmentation_type="noise", factor=0.005):
    if augmentation_type == "time_stretch":
        y_aug = librosa.effects.time_stretch(y, rate=1.1)
    elif augmentation_type == "pitch_shift":
        y_aug = librosa.effects.pitch_shift(y, sr=sr, n_steps=4)
    elif augmentation_type == "noise":
        noise = np.random.randn(len(y))
        y_aug = y + factor * noise
    else:
        y_aug = y
    return y_aug


# Feature extraction function including additional features
def extract_features(file_path, augment=False, augmentation_type="noise"):
    y, sr = librosa.load(file_path, sr=None)

    # Apply augmentation if specified
    if augment:
        y = augment_audio(y, sr, augmentation_type=augmentation_type)

    # Extract features...
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
    spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
    chroma_stft = librosa.feature.chroma_stft(y=y, sr=sr)

    # Concatenate all features into a single vector
    features = np.concatenate((mfccs, spectral_centroid, chroma_stft), axis=0)
    features = np.mean(features.T, axis=0)

    return features


# Data preparation function that calls extract_features
def get_data(dataset_path, test_size=0.25, augment_data=False):
    features = []
    labels = []

    # Iterate over each emotion folder
    for emotion in os.listdir(dataset_path):
        if emotion not in ['ANG', 'DIS', 'FEA', 'HAP', 'NEU', 'SAD']:
            continue

        emotion_path = os.path.join(dataset_path, emotion)
        for file_name in os.listdir(emotion_path):
            if not file_name.lower().endswith('.wav'):
                continue

            file_path = os.path.join(emotion_path, file_name)
            feature_vector = extract_features(file_path, augment=augment_data)

            features.append(feature_vector)
            labels.append(emotion)

    # Convert labels to numeric and one-hot encode
    unique_labels = np.unique(labels)
    label_to_int = {label: i for i, label in enumerate(unique_labels)}
    int_labels = np.array([label_to_int[label] for label in labels])
    one_hot_labels = to_categorical(int_labels)

    # Split and normalize the dataset
    X_train, X_test, y_train, y_test = train_test_split(np.array(features), one_hot_labels, test_size=test_size,
                                                        random_state=42)
    scaler = StandardScaler().fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)

    return X_train, X_test, y_train, y_test
