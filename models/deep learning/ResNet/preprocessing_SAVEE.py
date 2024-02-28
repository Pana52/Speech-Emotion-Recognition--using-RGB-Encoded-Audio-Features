import librosa
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

# Constants
SAMPLE_RATE = 22050
MFCC_NUM = 13
TRACK_DURATION = 3  # Measured in seconds

# Emotion labels mapping
EMOTIONS = {
    'an': 'anger',
    'di': 'disgust',
    'fe': 'fear',
    'ha': 'happiness',
    'ne': 'neutral',
    'sa': 'sadness',
    'su': 'surprise'
}


def extract_features(file_path, sr=SAMPLE_RATE, duration=TRACK_DURATION):
    audio, sample_rate = librosa.load(file_path, sr=sr, duration=duration)
    if len(audio) < sr * duration:
        audio = np.pad(audio, (0, sr * duration - len(audio)), 'constant')
    mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=MFCC_NUM)
    mfccs_processed = np.mean(mfccs.T, axis=0)
    return mfccs_processed


def parse_filename(filename):
    parts = filename.split('_')
    speaker, emotion_code = parts[0], parts[1][:2]  # Adjusted to correctly handle two-letter emotion codes
    emotion = EMOTIONS.get(emotion_code, 'unknown')
    return speaker, emotion


def load_data(dataset_path):
    features, labels = [], []
    for file in os.listdir(dataset_path):
        if file.endswith(".wav"):
            speaker, emotion = parse_filename(file)
            feature = extract_features(os.path.join(dataset_path, file))
            if feature.size == 0:
                continue
            features.append(feature)
            labels.append(emotion)

    features = np.array(features)
    labels = np.array(labels)

    # Debugging: Check label distribution before encoding
    print("Label distribution before encoding:", np.unique(labels, return_counts=True))

    le = LabelEncoder()
    labels_encoded = le.fit_transform(labels)

    # Debugging: Check label distribution after encoding
    print("Label distribution after encoding:", np.unique(labels_encoded, return_counts=True))

    X_train, X_test, y_train, y_test = train_test_split(features, labels_encoded, test_size=0.2, random_state=42,
                                                        stratify=labels_encoded)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    return X_train, X_test, y_train, y_test
