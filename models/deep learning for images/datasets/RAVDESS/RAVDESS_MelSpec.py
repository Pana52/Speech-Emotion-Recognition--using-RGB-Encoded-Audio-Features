import os
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

# Constants
SAMPLE_RATE = 48000
N_MELS = 256
HOP_LENGTH = 1024
N_FFT = 2048
IMAGE_SIZE = (256, 256)

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
                                 n_fft=N_FFT, image_size=IMAGE_SIZE):
    try:
        # Load the audio file
        audio, sample_rate = librosa.load(file_path, sr=sr)

        # Compute the Mel-spectrogram
        S = librosa.feature.melspectrogram(y=audio, sr=sample_rate, n_mels=n_mels, hop_length=hop_length, n_fft=n_fft)

        # Convert to dB
        S_dB = librosa.power_to_db(S, ref=np.max)

        # Define the figure size based on the desired image size and dpi
        dpi = 300
        fig_width = image_size[0] / dpi
        fig_height = image_size[1] / dpi
        plt.figure(figsize=(fig_width, fig_height), dpi=dpi)

        # Plot the Mel-spectrogram using a visually distinctive colormap
        librosa.display.specshow(S_dB, sr=sample_rate, hop_length=hop_length, x_axis='time', y_axis='mel', cmap='inferno')
        plt.axis('off')

        # Generate a temporary save path
        temp_save_path = f"{output_dir}/{file_name}.png"

        # Save the figure
        plt.savefig(temp_save_path, bbox_inches='tight', pad_inches=0)
        plt.close()

        # Open the image and resize it if necessary
        img = Image.open(temp_save_path)
        if img.size != image_size:
            img_resized = img.resize(image_size, Image.Resampling.LANCZOS)
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
    output_path = "C:/Users/Pana/Desktop/Northumbria/Final Year/Individual Computing Project " \
                 "KV6003BNN01/Speech-Emotion-Recognition---Audio-Dataset/models/deep learning for " \
                 "images/datasets/RAVDESS\Mel-Spectrograms/MelSpec_256x256/"
    main(dataset_path, output_path)
