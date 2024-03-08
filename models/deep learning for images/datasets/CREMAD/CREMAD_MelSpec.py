import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import os

# Constants
SAMPLE_RATE = 22050
N_MELS = 128  # Adjust this if needed
HOP_LENGTH = 512
N_FFT = 2048
IMAGE_SIZE = (224, 224)  # Target size for your images, adjust as needed


def save_mel_spectrogram(file_path, output_dir, file_name, sr=SAMPLE_RATE, n_mels=N_MELS, hop_length=HOP_LENGTH,
                         n_fft=N_FFT):
    """
    Generates a Mel spectrogram from an audio file and saves it as an image.
    """
    try:
        audio, sample_rate = librosa.load(file_path, sr=sr)
        S = librosa.feature.melspectrogram(y=audio, sr=sample_rate, n_mels=n_mels, hop_length=hop_length, n_fft=n_fft)
        S_DB = librosa.power_to_db(S, ref=np.max)
        plt.figure(figsize=(IMAGE_SIZE[0] / 100, IMAGE_SIZE[1] / 100),
                   dpi=100)  # Assuming 100 dpi to match the target size
        librosa.display.specshow(S_DB, sr=sample_rate, hop_length=hop_length, x_axis='time', y_axis='mel')
        plt.axis('off')
        plt.savefig(f"{output_dir}/{file_name}.png", bbox_inches='tight', pad_inches=0)
        plt.close()
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
    output_dir = '/models/deep learning for images/models/CNN/CREMAD ' \
                 'MELSPEC/'
    process_audio_files(data_path, output_dir)
