import os

import librosa
import numpy as np
import pandas as pd
import audformat
import audeer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Define the mappings based on the provided dataset information
EMOTION_MAPPING = {
    "W": "anger",
    "L": "boredom",
    "E": "disgust",
    "A": "fear",
    "F": "happiness",
    "T": "sadness",
    "N": "neutral",
}


def read_emotion_file(src_dir):
    """
    Reads the emotion file and returns a DataFrame with recognition rates.

    Args:
        src_dir (str): Source directory where the erkennung.txt file is located.

    Returns:
        pandas.DataFrame: DataFrame with recognition rates.
    """
    erkennung_path = os.path.join(src_dir, "erkennung.txt")
    erkennung_df = pd.read_csv(
        erkennung_path,
        sep="\s+",
        encoding="latin1",
        index_col="Nr.",
        usecols=["Satz", "erkannt", "natuerlich"],
        decimal=",",
    )
    erkennung_df['erkannt'] = erkennung_df['erkannt'].str.replace(',', '.').astype(float)
    erkennung_df['natuerlich'] = erkennung_df['natuerlich'].str.replace(',', '.').astype(float)

    # Normalize the confidence scores to a range of 0-1
    erkennung_df['confidence'] = erkennung_df['erkannt'] / 100.0

    return erkennung_df[['Satz', 'confidence']]


def create_audformat_db(src_dir, files, emotions, confidences, speakers, transcriptions):
    """
    Creates an audformat database with the provided information.

    Args:
        src_dir (str): Source directory containing the dataset.
        files (list): List of file paths.
        emotions (list): List of emotions corresponding to files.
        confidences (list): List of confidence scores.
        speakers (list): List of speaker IDs.
        transcriptions (list): List of transcriptions.

    Returns:
        audformat.Database: The created audformat database.
    """
    db = audformat.Database(
        name="emodb",
        source="Technical University Berlin",
        usage=audformat.define.Usage.UNRESTRICTED,
        languages=[audformat.utils.map_language('de')],
        description=(
            "Berlin Database of Emotional Speech. "
            "Recordings of 10 actors expressing 7 different emotions. "
            "All utterances were recorded at a sampling rate of 16 kHz."
        )
    )

    # Define schemes
    db.schemes['emotion'] = audformat.Scheme(
        labels=list(EMOTION_MAPPING.values()),
        description="Emotion expressed in the recording."
    )
    db.schemes['confidence'] = audformat.Scheme(
        dtype='float',
        minimum=0.0,
        maximum=1.0,
        description="Confidence level of the emotion label."
    )
    db.schemes['speaker'] = audformat.Scheme(
        dtype='int',
        description="Identifier of the speaker."
    )
    db.schemes['transcription'] = audformat.Scheme(
        dtype='str',
        description="Transcription of the spoken sentence."
    )

    # Add tables
    db['files'] = audformat.Table(audformat.filewise_index(files))
    db['files']['speaker'] = audformat.Column(scheme_id='speaker')
    db['files']['speaker'].set(speakers)
    db['files']['emotion'] = audformat.Column(scheme_id='emotion')
    db['files']['emotion'].set(emotions)
    db['files']['confidence'] = audformat.Column(scheme_id='confidence')
    db['files']['confidence'].set(confidences)
    db['files']['transcription'] = audformat.Column(scheme_id='transcription')
    db['files']['transcription'].set(transcriptions)

    return db


def extract_features(file_path, sr=22050, augment=True, duration=3):
    audio, sample_rate = librosa.load(file_path, sr=sr, duration=duration)
    # Ensuring minimum audio length
    if len(audio) < sr * duration:
        pad_len = sr * duration - len(audio)
        audio = np.pad(audio, (0, pad_len), 'constant')

    # Feature extraction
    # Adjust these feature extraction steps based on the features you want to include
    mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
    mfccs_processed = np.mean(mfccs.T, axis=0)

    # mel_spec = librosa.feature.melspectrogram(y=audio, sr=sr)
    # mel_spec_processed = np.mean(librosa.power_to_db(mel_spec), axis=1)

    # chroma = librosa.feature.chroma_stft(y=audio, sr=sr)
    # chroma_processed = np.mean(chroma.T, axis=0)

    # spectral_contrast = librosa.feature.spectral_contrast(y=audio, sr=sr)
    # spectral_contrast_processed = np.mean(spectral_contrast, axis=1)

    # Combine all features
    features = np.hstack(mfccs_processed)
    # features = np.hstack(mel_spec_processed)

    return features


def load_data(data_path):
    """
    Load the data, extract features, preprocess them, and split into train and test sets.

    Args:
        data_path (str): Path to the dataset directory.

    Returns:
        Tuple of numpy.ndarrays: (X_train, X_test, y_train, y_test)
    """
    # List all WAV files
    wav_dir = os.path.join(data_path, 'wav')
    files = [os.path.join(wav_dir, f) for f in sorted(os.listdir(wav_dir)) if f.endswith('.wav')]

    # Extract features for each WAV file
    features = [extract_features(f) for f in files]
    # Normalize features
    scaler = StandardScaler()
    features = scaler.fit_transform(np.array(features, dtype=float))

    # Get emotion labels from filenames
    emotions = [EMOTION_MAPPING[os.path.splitext(os.path.basename(f))[0][5]] for f in files]
    # Convert emotion labels to numeric
    y = np.array([list(EMOTION_MAPPING.values()).index(e) for e in emotions])

    # Split the dataset
    X_train, X_test, y_train, y_test = train_test_split(features, y, test_size=0.2, random_state=42)

    return X_train, X_test, y_train, y_test
