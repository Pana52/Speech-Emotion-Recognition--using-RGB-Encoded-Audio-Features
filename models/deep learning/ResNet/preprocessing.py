import os
import librosa
import numpy as np
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
import torch

# Path to the CREMA-D dataset
DATASET_PATH = 'C:/Users/Pana/Desktop/Northumbria/Final Year/Individual Computing Project KV6003BNN01/datasets/CREMAD/'
# Emotion categories
EMOTIONS = ["ANG", "DIS", "FEA", "HAP", "NEU", "SAD"]


def add_random_noise(signal, noise_level=0.005):
    noise = np.random.randn(len(signal))
    augmented_signal = signal + noise_level * noise
    # Ensure the signal is in the same range
    augmented_signal = augmented_signal.astype(type(signal[0]))
    return augmented_signal


def pad_mfcc(mfcc, max_len=216):
    """
    Pads or truncates the MFCC array to a fixed length.

    Parameters:
    - mfcc: The MFCC array to pad.
    - max_len: The target length to pad or truncate to.

    Returns:
    - The padded or truncated MFCC array.
    """
    pad_width = max_len - mfcc.shape[1]
    if pad_width > 0:
        # Pad the array if it is shorter than the max_len
        mfcc = np.pad(mfcc, pad_width=((0, 0), (0, pad_width)), mode='constant')
    else:
        # Truncate the array if it is longer than the max_len
        mfcc = mfcc[:, :max_len]
    return mfcc


def load_data(dataset_path=DATASET_PATH, emotions=EMOTIONS, max_len=216, augment=False, augment_probability=0.5):
    X, y = [], []
    for emotion in emotions:
        emotion_path = os.path.join(dataset_path, emotion)
        for filename in os.listdir(emotion_path):
            if not filename.lower().endswith(('.wav', '.mp3', '.flac')):
                continue
            file_path = os.path.join(emotion_path, filename)
            try:
                signal, sr = librosa.load(file_path)
                # Apply augmentation with a certain probability
                if augment and np.random.rand() < augment_probability:
                    signal = add_random_noise(signal)
                mfcc = librosa.feature.mfcc(y=signal, sr=sr, n_mfcc=13)
                mfcc = pad_mfcc(mfcc, max_len=max_len)
                X.append(mfcc)
                y.append(emotions.index(emotion))
            except Exception as e:
                print(f"Error loading {file_path}: {e}")
    X = np.stack(X)
    y = np.array(y)
    return X, y


def preprocess_data(X, y, test_size=0.2, validation_size=0.2):
    # Split data into training and testing
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
    # Split training data into training and validation
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=validation_size, random_state=42)
    return X_train, X_val, X_test, y_train, y_val, y_test


class AudioDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        # Reshape the MFCC features to add a channel dimension
        mfcc_features = np.expand_dims(self.X[idx], axis=0)
        return torch.tensor(mfcc_features, dtype=torch.float32), torch.tensor(self.y[idx], dtype=torch.int64)


def get_dataloaders(batch_size=32, max_len=216, augment=False):
    X, y = load_data(DATASET_PATH, max_len=max_len, augment=augment)
    X_train, X_val, X_test, y_train, y_val, y_test = preprocess_data(X, y)

    # Convert datasets into DataLoader
    train_loader = DataLoader(AudioDataset(X_train, y_train), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(AudioDataset(X_val, y_val), batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(AudioDataset(X_test, y_test), batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader
