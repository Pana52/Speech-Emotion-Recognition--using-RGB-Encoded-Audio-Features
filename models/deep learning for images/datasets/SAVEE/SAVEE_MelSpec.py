import librosa
import librosa.display
import matplotlib.pyplot as plt
from PIL import Image
import os
import numpy as np

# Constants
SAMPLE_RATE = 44100
N_MELS = 256
HOP_LENGTH = 128
N_FFT = 4096
IMAGE_SIZE = (512, 512)

# Emotion labels
EMOTIONS = {
    'an': 'anger',
    'di': 'disgust',
    'fe': 'fear',
    'ha': 'happiness',
    'ne': 'neutral',
    'sa': 'sadness',
    'su': 'surprise'
}


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


def main():
    dataset_dir = 'C:/Users/Pana/Desktop/Northumbria/Final Year/Individual Computing Project ' \
                  'KV6003BNN01/datasets/SAVEE/'  # Update this to your dataset path
    base_output_dir = '/models/deep learning for images/datasets/SAVEE/Mel-Spectrograms/MELSPEC_512x512/'

    # Iterate through each file in the dataset directory
    for file in os.listdir(dataset_dir):
        if file.endswith('.wav'):
            file_path = os.path.join(dataset_dir, file)
            emotion_code = file.split('_')[1][:2]  # Assumes the format "XX_emYY.wav"
            emotion_label = EMOTIONS.get(emotion_code, 'unknown')

            output_dir = os.path.join(base_output_dir, emotion_label)
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)

            file_name = file.replace('.wav', '')
            print(f"Processing {file_path}...")
            create_mel_spectrogram(file_path, output_dir, file_name)
            print(f"Saved Mel-Spectrogram to {output_dir}/{file_name}.png")


if __name__ == '__main__':
    main()
