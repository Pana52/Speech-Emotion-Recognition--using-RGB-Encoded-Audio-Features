import librosa
import numpy as np
import os
from librosa.display import specshow
import matplotlib.pyplot as plt


def load_audio_files(directory_path, sample_rate=22050):
    """
    Loads audio files from a specified directory.

    Parameters:
    - directory_path: str, path to the directory containing audio files.
    - sample_rate: int, sampling rate to use when loading the audio.

    Returns:
    - A dictionary mapping file names to their loaded data and sample rate.
    """
    audio_files = {}
    for filename in os.listdir(directory_path):
        if filename.endswith(".wav"):
            file_path = os.path.join(directory_path, filename)
            data, sr = librosa.load(file_path, sr=sample_rate)
            audio_files[filename] = (data, sr)
    return audio_files


def audio_to_melspectrogram(audio_data, sample_rate=22050, n_fft=2048, hop_length=512, n_mels=128):
    """
    Converts an audio time series to a Mel-Spectrogram.

    Parameters:
    - audio_data: np.array, audio time series.
    - sample_rate: int, sampling rate of the audio_data.
    - n_fft: int, length of the FFT window.
    - hop_length: int, number of samples between successive frames.
    - n_mels: int, number of Mel bands to generate.

    Returns:
    - M: np.ndarray [shape=(n_mels, t)], Mel spectrogram.
    """
    S = librosa.feature.melspectrogram(y=audio_data, sr=sample_rate, n_fft=n_fft,
                                       hop_length=hop_length, n_mels=n_mels)
    M = librosa.power_to_db(S, ref=np.max)
    return M


def augment_audio(audio_data, sample_rate=22050, augmentation_type="noise"):
    """
    Applies data augmentation to audio data. Currently supports noise injection.

    Parameters:
    - audio_data: np.array, audio time series.
    - sample_rate: int, sampling rate of the audio_data.
    - augmentation_type: str, type of augmentation to apply. Currently only 'noise'.

    Returns:
    - augmented_audio: np.array, augmented audio data.
    """
    if augmentation_type == "noise":
        noise_amp = 0.005 * np.random.uniform() * np.amax(audio_data)
        augmented_audio = audio_data + noise_amp * np.random.normal(size=audio_data.shape[0])
    else:
        augmented_audio = audio_data  # No augmentation applied

    return augmented_audio
