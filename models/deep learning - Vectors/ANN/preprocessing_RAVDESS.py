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


# Filename parsing and emotion mapping
def parse_filename(filename):
    parts = filename.split('-')
    emotion_code = parts[2]
    return emotion_code


def emotion_label(emotion_code):
    emotions = {
        '01': 'neutral', '02': 'calm', '03': 'happy', '04': 'sad',
        '05': 'angry', '06': 'fearful', '07': 'disgust', '08': 'surprised'
    }
    return emotions.get(emotion_code, None)


# Extract features with optional augmentation
def extract_features(file_path, sr=22050, augment=True, duration=3):
    audio, sample_rate = librosa.load(file_path, sr=sr, duration=duration)
    if len(audio) < sr * duration:
        pad_len = sr * duration - len(audio)
        audio = np.pad(audio, (0, pad_len), 'constant')
    # mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=MFCC_NUM)
    # mfccs_processed = np.mean(mfccs.T, axis=0)
    # features = np.hstack(mfccs_processed)
    mel_spec = librosa.feature.melspectrogram(y=audio, sr=sr)
    mel_spec_processed = np.mean(librosa.power_to_db(mel_spec), axis=1)
    features = np.hstack(mel_spec_processed)

    return features


def load_data(data_path):
    labels = []
    features = []
    file_count = 0  # Debugging: Count how many files are processed

    # Iterate over all actor directories
    for actor_dir in os.listdir(data_path):
        actor_path = os.path.join(data_path, actor_dir)
        if os.path.isdir(actor_path):  # Ensure it's a directory
            for filename in os.listdir(actor_path):
                if filename.endswith('.wav'):
                    file_path = os.path.join(actor_path, filename)
                    emotion_code = parse_filename(filename)
                    emotion = emotion_label(emotion_code)
                    if emotion:  # Ensure emotion is known
                        features.append(extract_features(file_path))
                        labels.append(emotion)
                        file_count += 1  # Increment file count
                    else:
                        print(f"Unknown emotion for file: {filename}")  # Debugging: Print unknown emotion files
                else:
                    print(f"Skipped non-audio file: {filename}")  # Debugging: Print skipped files

    print(f"Total files processed: {file_count}")  # Debugging: Print total files processed

    if file_count == 0:
        print("No files were processed. Check the data path and file format.")
        return None, None, None, None

    le = LabelEncoder()
    labels_encoded = le.fit_transform(labels)
    X_train, X_test, y_train, y_test = train_test_split(np.array(features), labels_encoded, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    return X_train, X_test, y_train, y_test
