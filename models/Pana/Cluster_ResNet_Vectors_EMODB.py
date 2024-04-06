# Full adapted code for processing audio files, including feature extraction and clustering

# Import necessary libraries
import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from keras.models import Model
from keras.layers import Input, Dense
from keras.optimizers import Adam
from keras.utils import to_categorical
from keras.callbacks import EarlyStopping
from sklearn.metrics import classification_report
import librosa

# Constants
DATA_DIR = "C:/Users/Pana/Desktop/Northumbria/Final Year/Individual Computing Project KV6003BNN01/datasets/audio/"
AUDIO_SUBFOLDER = 'EMODB'
EMOTIONS = ['anger', 'boredom', 'disgust', 'fear', 'happiness', 'neutral', 'sadness']
NUM_CLASSES = len(EMOTIONS)
SAMPLE_RATE = 48000
N_MFCC = 13
N_FFT = 1024
N_CLUSTERS = 10


# Function to extract features from an audio file
def extract_audio_features(audio_path, n_fft=N_FFT):
    y, sr = librosa.load(audio_path, sr=SAMPLE_RATE)

    # Ensure signal is at least n_fft in length
    if len(y) < n_fft:
        y = np.pad(y, pad_width=(0, n_fft - len(y)), mode='constant')

    # Now proceed with feature extraction, safely using n_fft
    try:
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC, n_fft=n_fft)
        chroma = librosa.feature.chroma_stft(y=y, sr=sr, n_fft=n_fft)
        mel = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=n_fft)
        contrast = librosa.feature.spectral_contrast(y=y, sr=sr, n_fft=n_fft)
        tonnetz = librosa.feature.tonnetz(y=librosa.effects.harmonic(y), sr=sr)
    except ValueError as e:
        print(f"Error processing file {audio_path}: {e}")
        return np.zeros((N_MFCC,))  # Return an array of zeros if there's an error

    features = np.hstack([np.mean(feat, axis=1) for feat in [mfccs, chroma, mel, contrast, tonnetz]])
    return features


# Function to apply clustering on the extracted features
def apply_clustering(features):
    kmeans = KMeans(n_clusters=N_CLUSTERS, random_state=42)
    cluster_labels = kmeans.fit_predict(features)
    return cluster_labels, kmeans.cluster_centers_


# Function to build a classification model
def build_classification_model(num_classes, input_shape):
    input_layer = Input(shape=input_shape)
    x = Dense(512, activation='relu')(input_layer)
    x = Dense(512, activation='relu')(x)
    predictions = Dense(num_classes, activation='softmax')(x)
    model = Model(inputs=input_layer, outputs=predictions)
    model.compile(optimizer=Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])
    return model


# Function to load dataset and extract audio features
def load_dataset_and_extract_audio_features(data_dir):
    features = []
    labels = []

    for emotion in EMOTIONS:
        print(f"Processing {emotion} audio clips...")
        emotion_audio_path = os.path.join(data_dir, AUDIO_SUBFOLDER, emotion)
        for audio_file in os.listdir(emotion_audio_path):
            full_audio_path = os.path.join(emotion_audio_path, audio_file)
            audio_features = extract_audio_features(full_audio_path)
            features.append(audio_features)
            labels.append(EMOTIONS.index(emotion))

    features = np.array(features)
    labels = to_categorical(labels, num_classes=NUM_CLASSES)

    return features, labels


# Main function to execute the process
def main(data_dir):
    # Load dataset and extract audio features
    features, labels = load_dataset_and_extract_audio_features(data_dir)

    # Normalize features
    scaler = StandardScaler()
    features = scaler.fit_transform(features)

    # Apply clustering
    cluster_labels, _ = apply_clustering(features)

    # Proceed with classification
    input_shape = (features.shape[1],)  # Adjust based on the extracted features
    classification_model = build_classification_model(NUM_CLASSES, input_shape=input_shape)
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

    # Early Stopping
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, verbose=1, restore_best_weights=True)

    # Train the model
    classification_model.fit(X_train, y_train, batch_size=32, epochs=100, validation_data=(X_test, y_test),
                             callbacks=[early_stopping])

    # Predictions for the test set
    y_pred = classification_model.predict(X_test)
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_true_classes = np.argmax(y_test, axis=1)

    # Classification report
    print(classification_report(y_true_classes, y_pred_classes, target_names=EMOTIONS))


if __name__ == "__main__":
    main(DATA_DIR)

# Note: This script assumes you have an appropriate dataset and the 'librosa' library installed.
# You may need to install librosa using pip (pip install librosa) before running this code.
