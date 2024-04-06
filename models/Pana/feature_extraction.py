import os
import librosa
import numpy as np
import matplotlib.pyplot as plt
from librosa.display import specshow
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

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


def create_mel_spectrogram(audio, sr=SAMPLE_RATE, save_path='mel_spectrogram.png', n_mels=128, hop_length=1024, n_fft=2048):
    # Corrected function call with keyword arguments
    S = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=n_mels, hop_length=hop_length, n_fft=n_fft)
    S_DB = librosa.power_to_db(S, ref=np.max)
    plt.figure(figsize=(10, 4))
    specshow(S_DB, sr=sr, hop_length=hop_length, x_axis='time', y_axis='mel')
    plt.axis('off')
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
    plt.close()


def apply_clustering(features, n_clusters=N_CLUSTERS):
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    features_flat = np.array(features).reshape(len(features), -1)
    enhanced_features = kmeans.fit_transform(features_flat)
    return enhanced_features


def process_dataset(dataset_path=DATASET_PATH, emotions=EMOTIONS, test_size=0.2, validation_size=0.2):
    features, labels = [], []
    audio_base_path = os.path.join(dataset_path, "Audio")
    image_base_path = os.path.join(dataset_path, "Image")

    for emotion in emotions:
        audio_emotion_path = os.path.join(audio_base_path, emotion)
        image_emotion_path = os.path.join(image_base_path, emotion)
        if not os.path.exists(image_emotion_path):
            os.makedirs(image_emotion_path)

        for filename in os.listdir(audio_emotion_path):
            file_path = os.path.join(audio_emotion_path, filename)
            audio = load_and_preprocess_audio(file_path)
            mfcc = extract_mfcc(audio)

            mel_spec_filename = filename.replace(".wav", ".png")
            mel_spec_path = os.path.join(image_emotion_path, mel_spec_filename)

            create_mel_spectrogram(audio, save_path=mel_spec_path)
            features.append((mfcc, mel_spec_path))
            labels.append(emotion)

    mfcc_features = [feature[0] for feature in features]  # Extracting MFCCs for clustering
    enhanced_features = apply_clustering(mfcc_features, n_clusters=N_CLUSTERS)
    label_to_index = {emotion: index for index, emotion in enumerate(emotions)}
    numeric_labels = [label_to_index[label] for label in labels]
    X_train, X_test, y_train, y_test = train_test_split(enhanced_features, numeric_labels, test_size=test_size,
                                                        stratify=numeric_labels, random_state=42)
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
