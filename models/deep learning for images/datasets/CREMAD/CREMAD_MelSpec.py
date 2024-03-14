import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import os
from PIL import Image

# Constants
SAMPLE_RATE = 44100
N_MELS = 256
HOP_LENGTH = 128
N_FFT = 4096
IMAGE_SIZE = (512, 512)


def save_mel_spectrogram(file_path, output_dir, file_name, sr=SAMPLE_RATE, n_mels=N_MELS, hop_length=HOP_LENGTH,
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


def process_audio_files(data_path, output_dir):
    """
    Processes all audio files in the specified directory, generating and saving Mel spectrogram images.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    emotions = ['ANG', 'DIS', 'FEA', 'HAP', 'NEU', 'SAD']
    for emotion in emotions:
        emotion_path = os.path.join(data_path, emotion)
        emotion_output_dir = os.path.join(output_dir, emotion)
        if not os.path.exists(emotion_output_dir):
            os.makedirs(emotion_output_dir)

        for filename in os.listdir(emotion_path):
            if filename.endswith('.wav'):
                file_path = os.path.join(emotion_path, filename)
                save_mel_spectrogram(file_path, emotion_output_dir, filename.split('.')[0])


if __name__ == "__main__":
    # Define the path to your CREMA-D dataset and the output directory for the spectrogram images
    data_path = 'C:/Users/Pana/Desktop/Northumbria/Final Year/Individual Computing Project KV6003BNN01/datasets/CREMAD/'
    output_dir = '/models/deep learning for images/datasets/CREMAD/Mel-Spectrograms/MELSPEC_512x512/'
    process_audio_files(data_path, output_dir)
