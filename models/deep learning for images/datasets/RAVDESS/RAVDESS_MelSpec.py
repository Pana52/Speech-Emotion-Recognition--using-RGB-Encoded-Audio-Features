import os
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

# Constants
SAMPLE_RATE = 44100
N_MELS = 256
HOP_LENGTH = 128
N_FFT = 4096
IMAGE_SIZE = (512, 512)

# RAVDESS emotion mapping
emotions = {
    '01': 'neutral',
    '02': 'calm',
    '03': 'happy',
    '04': 'sad',
    '05': 'angry',
    '06': 'fearful',
    '07': 'disgust',
    '08': 'surprised',
}


def parse_filename(filename):
    # RAVDESS filenames are structured as Modality-VocalChannel-Emotion-Intensity-Statement-Repetition-Actor.wav
    # For example: 03-01-06-01-02-01-12.wav
    parts = filename.split('.')[0].split('-')
    emotion_code = parts[2]
    return emotions[emotion_code]


def create_mel_spectrogram(file_path, output_dir, file_name, sr=SAMPLE_RATE, n_mels=N_MELS, hop_length=HOP_LENGTH,
                           n_fft=N_FFT):
    try:
        audio, sample_rate = librosa.load(file_path, sr=sr)
        S = librosa.feature.melspectrogram(y=audio, sr=sample_rate, n_mels=n_mels, hop_length=hop_length, n_fft=n_fft)
        S_DB = librosa.power_to_db(S, ref=np.max)

        # Create the figure
        plt.figure(figsize=(IMAGE_SIZE[0] / 100, IMAGE_SIZE[1] / 100), dpi=100)
        librosa.display.specshow(S_DB, sr=sample_rate, hop_length=hop_length, x_axis='time', y_axis='mel')
        plt.axis('off')

        # Temporary save path
        temp_save_path = f"{output_dir}/{file_name}.png"
        plt.savefig(temp_save_path, bbox_inches='tight', pad_inches=0)
        plt.close()

        # Resize the image to the desired output size using Image.Resampling.LANCZOS
        img = Image.open(temp_save_path)
        img_resized = img.resize(IMAGE_SIZE, Image.Resampling.LANCZOS)
        img_resized.save(temp_save_path)

    except Exception as e:
        print(f"Error processing {file_name}: {e}")


def main(dataset_path, output_path):
    for actor_id in os.listdir(dataset_path):
        actor_path = os.path.join(dataset_path, actor_id)
        if not os.path.isdir(actor_path):
            continue  # Skip any files, process only directories
        for filename in os.listdir(actor_path):
            if filename.endswith(".wav"):
                emotion = parse_filename(filename)
                audio_path = os.path.join(actor_path, filename)
                save_directory = os.path.join(output_path, emotion)
                os.makedirs(save_directory, exist_ok=True)
                file_name_without_ext = filename.replace(".wav", "")
                create_mel_spectrogram(audio_path, save_directory, file_name_without_ext)
                save_path = os.path.join(save_directory, file_name_without_ext + ".png")
                print(f"Mel-Spectrogram saved to {save_path}")


if __name__ == "__main__":
    dataset_path = "C:/Users/Pana/Desktop/Northumbria/Final Year/Individual Computing Project " \
                   "KV6003BNN01/datasets/RAVDESS/"  # Path to the dataset folder
    output_path = '/models/deep learning for images/datasets/RAVDESS/Mel-Spectrograms/MELSPEC_512x512/'  # Path where Mel-Spectrogram images will be saved
    main(dataset_path, output_path)
