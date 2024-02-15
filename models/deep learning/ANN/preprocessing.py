import librosa
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

# Constants
SAMPLE_RATE = 22050
MFCC_NUM = 13
TRACK_DURATION = 3  # Measured in seconds
SAMPLES_PER_TRACK = SAMPLE_RATE * TRACK_DURATION


def add_noise(data):
    noise_factor = 0.005
    noise = np.random.randn(len(data))
    augmented_data = data + noise_factor * noise
    # Cast back to same data type
    augmented_data = augmented_data.astype(type(data[0]))
    return augmented_data


def time_shift(data):
    shift_range = int(np.random.uniform(low=-5, high=5) * 1000)
    augmented_data = np.roll(data, shift_range)
    return augmented_data


def extract_features(file_path, augment=False):
    audio, sample_rate = librosa.load(file_path, sr=SAMPLE_RATE, mono=True)
    if len(audio) > SAMPLES_PER_TRACK:
        audio = audio[:SAMPLES_PER_TRACK]
    if augment:
        # Apply augmentations
        if np.random.uniform(0, 1) < 0.5:
            audio = add_noise(audio)
        if np.random.uniform(0, 1) < 0.5:
            audio = time_shift(audio)
    mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=MFCC_NUM)
    mfccs_processed = np.mean(mfccs.T, axis=0)
    return mfccs_processed


def load_data(data_path, augment=False):
    labels = []
    features = []

    emotions = ['ANG', 'DIS', 'FEA', 'HAP', 'NEU', 'SAD']
    for emotion in emotions:
        emotion_path = os.path.join(data_path, emotion)
        for filename in os.listdir(emotion_path):
            file_path = os.path.join(emotion_path, filename)
            if file_path.endswith('.wav'):
                features.append(extract_features(file_path, augment=augment))
                labels.append(emotion)
                if augment:
                    # Add augmented data
                    features.append(extract_features(file_path, augment=True))
                    labels.append(emotion)

    # Encode labels
    le = LabelEncoder()
    labels_encoded = le.fit_transform(labels)

    # Split the dataset
    X_train, X_test, y_train, y_test = train_test_split(np.array(features), labels_encoded, test_size=0.2,
                                                        random_state=42)

    # Scale features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    return X_train, X_test, y_train, y_test
