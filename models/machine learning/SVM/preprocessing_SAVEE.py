from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
import numpy as np
import os
import librosa

# Constants
SAMPLE_RATE = 22050
MFCC_NUM = 13
TRACK_DURATION = 3  # seconds
EXPECTED_MFCC_FEATURES = 308

# Emotion labels
EMOTIONS = {
    'an': 'anger',
    'di': 'disgust',
    'fe': 'fear',
    'ha': 'happiness',
    'ne': 'neutral',
    'sa': 'sadness',
    'su': 'surprise'
}


def extract_features(file_path, sr=SAMPLE_RATE, n_mfcc=MFCC_NUM, track_duration=TRACK_DURATION,
                     expected_mfcc_features=EXPECTED_MFCC_FEATURES):
    audio, sample_rate = librosa.load(file_path, sr=sr)
    if len(audio) < sr * track_duration:
        audio = np.pad(audio, (0, max(0, sr * track_duration - len(audio))), "constant")
    mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=n_mfcc)
    # Flatten the mfccs and ensure fixed size
    mfccs = mfccs.T.flatten()  # Flatten the array
    fixed_size_mfccs = np.zeros((expected_mfcc_features * n_mfcc,))
    # Copy the extracted features to the fixed-size array (truncate if necessary)
    fixed_size_mfccs[:min(mfccs.size, fixed_size_mfccs.size)] = mfccs[:min(mfccs.size, fixed_size_mfccs.size)]
    return fixed_size_mfccs


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

    X_train, X_test, y_train, y_test = train_test_split(features, labels_encoded, test_size=0.2, random_state=42, stratify=labels_encoded)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    return X_train, X_test, y_train, y_test