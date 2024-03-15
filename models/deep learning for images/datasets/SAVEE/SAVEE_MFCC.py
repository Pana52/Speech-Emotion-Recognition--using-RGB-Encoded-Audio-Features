import librosa
import librosa.display
import matplotlib.pyplot as plt
from PIL import Image
import os
import numpy as np

# Constants
SAMPLE_RATE = 48000
N_MFCC = 64
HOP_LENGTH = 2048
N_FFT = 4096
IMAGE_SIZE = (256, 256)

# Emotion labels remain the same
EMOTIONS = {
    'an': 'anger',
    'di': 'disgust',
    'fe': 'fear',
    'ha': 'happiness',
    'ne': 'neutral',
    'sa': 'sadness',
    'su': 'surprise'
}


def create_mfcc_image(file_path, output_dir, file_name, sr=SAMPLE_RATE, n_mfcc=N_MFCC, hop_length=HOP_LENGTH,
                      n_fft=N_FFT, image_size=IMAGE_SIZE):
    try:
        # Load the audio file
        audio, sample_rate = librosa.load(file_path, sr=sr)

        # Compute the MFCC
        mfcc = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=n_mfcc, hop_length=hop_length, n_fft=n_fft)

        # Standardize the MFCCs with an added epsilon for numerical stability
        epsilon = 1e-10
        mfcc_standardized = (mfcc - np.mean(mfcc, axis=1, keepdims=True)) / (
                    np.std(mfcc, axis=1, keepdims=True) + epsilon)

        # Define the figure size based on the desired image size and dpi
        dpi = 300
        fig_width = image_size[0] / dpi
        fig_height = image_size[1] / dpi
        plt.figure(figsize=(fig_width, fig_height), dpi=dpi)

        # Plot the standardized MFCC using a more visually distinctive colormap
        librosa.display.specshow(mfcc_standardized, sr=sample_rate, hop_length=hop_length, x_axis='time',
                                 cmap='inferno')
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
    base_output_dir = 'C:/Users/Pana/Desktop/Northumbria/Final Year/Individual Computing Project ' \
                      'KV6003BNN01/Speech-Emotion-Recognition---Audio-Dataset/models/deep learning for ' \
                      'images/datasets/SAVEE/MFCCs/MFCC_256x256/'

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
            create_mfcc_image(file_path, output_dir, file_name)
            print(f"Saved MFCC to {output_dir}/{file_name}.png")


if __name__ == '__main__':
    main()
