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
N_MELS = 40  # Number of Mel bands to generate


# Augmentation functions as previously defined
def add_noise(data, noise_level=0.005):
    noise = np.random.randn(len(data))
    augmented_data = data + noise_level * noise
    return augmented_data


def time_shift(data, sampling_rate, shift_max=0.2, shift_direction='both'):
    shift = np.random.randint(int(sampling_rate * shift_max))
    if shift_direction == 'right':
        shift = -shift
    elif shift_direction == 'both':
        direction = np.random.randint(0, 2)
        if direction == 1:
            shift = -shift
    augmented_data = np.roll(data, shift)
    if shift > 0:
        augmented_data[:shift] = 0
    else:
        augmented_data[shift:] = 0
    return augmented_data


def pitch_shift(data, sampling_rate, n_steps=0.7):
    return librosa.effects.pitch_shift(data, sr=sampling_rate, n_steps=n_steps)


def change_speed(data, speed_factor=0.9):
    return librosa.effects.time_stretch(data, rate=speed_factor)


# Extract features with optional augmentation
def extract_features(file_path, sr=22050, augment=False, duration=3):
    audio, sample_rate = librosa.load(file_path, sr=sr, duration=duration)
    # Ensuring minimum audio length
    if len(audio) < sr * duration:
        pad_len = sr * duration - len(audio)
        audio = np.pad(audio, (0, pad_len), 'constant')

    if augment:
        # Apply augmentations randomly
        if np.random.rand() < 0.5:
            audio = add_noise(audio)
        if np.random.rand() < 0.5:
            audio = time_shift(audio, sampling_rate=sr)
        if np.random.rand() < 0.5:
            audio = pitch_shift(audio, sampling_rate=sr)
        if np.random.rand() < 0.5:
            audio = change_speed(audio)

    # Feature extraction
    # Adjust these feature extraction steps based on the features you want to include
    mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
    chroma = librosa.feature.chroma_stft(y=audio, sr=sr)
    mel_spec = librosa.feature.melspectrogram(y=audio, sr=sr)
    spectral_contrast = librosa.feature.spectral_contrast(y=audio, sr=sr)

    mfccs_processed = np.mean(mfccs.T, axis=0)
    chroma_processed = np.mean(chroma.T, axis=0)
    mel_spec_processed = np.mean(librosa.power_to_db(mel_spec), axis=1)
    spectral_contrast_processed = np.mean(spectral_contrast, axis=1)

    # Combine all features
    features = np.hstack((mfccs_processed, chroma_processed, mel_spec_processed, spectral_contrast_processed))
    return features


def load_data(data_path):
    labels = []
    features = []

    emotions = ['ANG', 'DIS', 'FEA', 'HAP', 'NEU', 'SAD']
    for emotion in emotions:
        emotion_path = os.path.join(data_path, emotion)
        for filename in os.listdir(emotion_path):
            file_path = os.path.join(emotion_path, filename)
            if file_path.endswith('.wav'):
                features.append(extract_features(file_path))
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
