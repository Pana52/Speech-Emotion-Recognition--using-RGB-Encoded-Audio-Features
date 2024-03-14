import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import os
from PIL import Image

# Constants
SAMPLE_RATE = 44100
N_MFCC = 40
HOP_LENGTH = 128
N_FFT = 8192
IMAGE_SIZE = (512, 512)

# EMODB emotion mapping remains the same
emotion_mapping = {
    "W": "anger",
    "L": "boredom",
    "E": "disgust",
    "A": "fear",
    "F": "happiness",
    "T": "sadness",
    "N": "neutral",
}


def create_mfcc_image(file_path, output_dir, file_name, sr=SAMPLE_RATE, n_mfcc=N_MFCC, hop_length=HOP_LENGTH,
                      n_fft=N_FFT, image_size=IMAGE_SIZE):
    try:
        # Load the audio file
        audio, sample_rate = librosa.load(file_path, sr=sr)

        # Compute the MFCC
        mfcc = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=n_mfcc, hop_length=hop_length, n_fft=n_fft)

        # Normalize the MFCCs
        mfcc_normalized = (mfcc - np.min(mfcc)) / (np.max(mfcc) - np.min(mfcc))

        # Define the figure size based on the desired image size and dpi
        dpi = 100
        fig_width = image_size[0] / dpi
        fig_height = image_size[1] / dpi
        plt.figure(figsize=(fig_width, fig_height), dpi=dpi)

        # Plot the normalized MFCC
        librosa.display.specshow(mfcc_normalized, sr=sample_rate, hop_length=hop_length, x_axis='time', cmap='viridis')
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


def process_audio_files(data_path, output_dir):
    # Adjust data_path to specifically target the 'wav' subdirectory
    data_path = os.path.join(data_path, 'wav')

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Check if the modified data_path exists and contains .wav files
    if not os.path.exists(data_path) or not os.listdir(data_path):
        print(f"No .wav files found in {data_path}. Please check the directory.")
        return

    for filename in os.listdir(data_path):
        if filename.endswith('.wav'):
            # Extract emotion code from the filename
            emotion_code = filename[5]
            emotion = emotion_mapping.get(emotion_code, 'unknown')

            if emotion != 'unknown':
                emotion_output_dir = os.path.join(output_dir, emotion)
                if not os.path.exists(emotion_output_dir):
                    os.makedirs(emotion_output_dir)

                file_path = os.path.join(data_path, filename)
                create_mfcc_image(file_path, emotion_output_dir, filename.split('.')[0])
            else:
                print(f"Unknown emotion for file {filename}. Skipping.")
        else:
            print(f"Non-audio file found: {filename}. Skipping.")


if __name__ == "__main__":
    # Define the path to your EMODB dataset and the output directory for the spectrogram images
    data_path = 'C:/Users/Pana/Desktop/Northumbria/Final Year/Individual Computing Project KV6003BNN01/datasets/EMODB/'
    output_dir = 'C:/Users/Pana/Desktop/Northumbria/Final Year/Individual Computing Project ' \
                 'KV6003BNN01/Speech-Emotion-Recognition---Audio-Dataset/models/deep learning for ' \
                 'images/datasets/EMODB/MFCCs/MFCC_512x512/'

    print(f"Data path: {data_path}")
    print(f"Output directory: {output_dir}")

    process_audio_files(data_path, output_dir)
