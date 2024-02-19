import os
import numpy as np
import librosa
from sklearn.model_selection import train_test_split
import tensorflow as tf


def add_noise(data, noise_factor=0.005):
    noise = np.random.randn(len(data))
    augmented_data = data + noise_factor * noise
    augmented_data = augmented_data.astype(type(data[0]))
    return augmented_data


def change_pitch(data, sampling_rate, pitch_factor=0.7):
    return librosa.effects.pitch_shift(data, sr=sampling_rate, n_steps=pitch_factor)


def load_data(data_dir, augment=False):
    X, y = [], []
    max_length = 0

    for emotion in ["ANG", "DIS", "FEA", "HAP", "NEU", "SAD"]:
        emotion_dir = os.path.join(data_dir, emotion)
        for filename in os.listdir(emotion_dir):
            if not filename.endswith('.wav'):
                continue
            file_path = os.path.join(emotion_dir, filename)
            signal, sr = librosa.load(file_path)

            if augment:
                signal = add_noise(signal)
                signal = change_pitch(signal, sr)

            mfccs = librosa.feature.mfcc(y=signal, sr=sr, n_mfcc=40)
            if mfccs.shape[1] > max_length:
                max_length = mfccs.shape[1]
            X.append(mfccs)
            y.append(emotion)

    X_padded = np.zeros((len(X), 40, max_length))
    for i, x in enumerate(X):
        X_padded[i, :, :x.shape[1]] = x

    emotions = ["ANG", "DIS", "FEA", "HAP", "NEU", "SAD"]
    y = [emotions.index(e) for e in y]
    y = tf.keras.utils.to_categorical(y)

    return np.array(X_padded), np.array(y)


def split_data(X, y, test_size=0.2, val_size=0.2):
    X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=val_size, random_state=42)
    return X_train, X_val, X_test, y_train, y_val, y_test
