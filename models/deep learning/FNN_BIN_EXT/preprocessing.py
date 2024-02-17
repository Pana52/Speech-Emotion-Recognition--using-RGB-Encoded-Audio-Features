import os
import librosa
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder


def load_data(data_directory):
    X, y = [], []
    label_encoder = LabelEncoder()
    for folder in os.listdir(data_directory):
        if folder in ["ANG", "DIS", "FEA", "HAP", "NEU", "SAD"]:
            folder_path = os.path.join(data_directory, folder)
            for file in os.listdir(folder_path):
                if not file.endswith('.wav'):  # Adjust based on your audio file extension
                    continue  # Skip non-audio files
                file_path = os.path.join(folder_path, file)
                features = extract_features(file_path)
                X.append(features)
                y.append(folder)
    # Convert labels to integers using LabelEncoder
    y_encoded = label_encoder.fit_transform(y)
    return np.array(X), np.array(y_encoded)


def extract_features(file_path, num_mfcc=13, n_fft=2048, hop_length=512):
    audio, sample_rate = librosa.load(file_path, sr=None)
    if len(audio) == 0:
        print(f"Empty audio file: {file_path}")
        return np.array([])  # Return empty array if audio file is empty

    mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=num_mfcc, n_fft=n_fft, hop_length=hop_length)

    # Ensure features have shape (1, 13)
    if mfccs.shape[0] != 1 or mfccs.shape[1] != 13:
        # Pad or truncate mfccs to ensure consistent shape
        if mfccs.shape[1] < 13:
            mfccs = np.pad(mfccs, ((0, 0), (0, 13 - mfccs.shape[1])), mode='constant')
        elif mfccs.shape[1] > 13:
            mfccs = mfccs[:, :13]

    # Reshape mfccs to (1, 13) if necessary
    if mfccs.shape != (1, 13):
        mfccs = mfccs.reshape(1, -1)

    return mfccs


def normalize_features(features):
    # Reshape features to 2D array if necessary
    if features.ndim > 2:
        features = features.reshape(features.shape[0], -1)

    scaler = StandardScaler()
    return scaler.fit_transform(features)


def encode_labels(labels):
    encoder = LabelEncoder()
    return encoder.fit_transform(labels)


def split_dataset(features, labels, test_size=0.2, random_state=42):
    return train_test_split(features, labels, test_size=test_size, random_state=random_state)
