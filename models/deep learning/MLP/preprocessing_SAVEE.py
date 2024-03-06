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


def extract_features(file_path, sr=22050, augment=True, duration=3):
    audio, sample_rate = librosa.load(file_path, sr=sr, duration=duration)
    # Ensuring minimum audio length
    if len(audio) < sr * duration:
        pad_len = sr * duration - len(audio)
        audio = np.pad(audio, (0, pad_len), 'constant')

    # Feature extraction
    # Adjust these feature extraction steps based on the features you want to include
    # mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
    # mfccs_processed = np.mean(mfccs.T, axis=0)

    mel_spec = librosa.feature.melspectrogram(y=audio, sr=sr)
    mel_spec_processed = np.mean(librosa.power_to_db(mel_spec), axis=1)

    # chroma = librosa.feature.chroma_stft(y=audio, sr=sr)
    # chroma_processed = np.mean(chroma.T, axis=0)

    # spectral_contrast = librosa.feature.spectral_contrast(y=audio, sr=sr)
    # spectral_contrast_processed = np.mean(spectral_contrast, axis=1)

    # Combine all features
    # features = np.hstack(mfccs_processed)
    features = np.hstack(mel_spec_processed)

    return features


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