import librosa
import librosa.display
import matplotlib.pyplot as plt
from PIL import Image
import os
import numpy as np

# Constants
SAMPLE_RATE = 48000
N_MELS = 256
HOP_LENGTH = 1024
N_FFT = 2048
IMAGE_SIZE = (256, 256)

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



def main():
    dataset_dir = 'C:/Users/Pana/Desktop/Northumbria/Final Year/Individual Computing Project ' \
                  'KV6003BNN01/datasets/SAVEE/'  # Update this to your dataset path
    base_output_dir = "C:/Users/Pana/Desktop/Northumbria/Final Year/Individual Computing Project " \
                 "KV6003BNN01/Speech-Emotion-Recognition---Audio-Dataset/models/deep learning for " \
                 "images/datasets/SAVEE\Mel-Spectrograms/MelSpec_256x256/"

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
