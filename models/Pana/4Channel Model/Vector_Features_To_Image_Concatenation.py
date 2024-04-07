# Let's outline the functions and structure needed for the process
import os
from PIL import Image
import numpy as np
import librosa
from sklearn.preprocessing import MinMaxScaler
from scipy.ndimage import zoom

DATASET_AUDIO = "C:/Users/Pana/Desktop/Northumbria/Final Year/Individual Computing Project " \
                "KV6003BNN01/datasets/Mixed/EMODB/Audio/"
OUTPUT_IMAGES = "C:/Users/Pana/Desktop/Northumbria/Final Year/Individual Computing Project " \
                "KV6003BNN01/datasets/Mixed/EMODB/3CF_Images/"
EMOTIONS = ['anger', 'boredom', 'disgust', 'fear', 'happiness', 'neutral', 'sadness']


# Define the feature extraction function
def extract_audio_features(audio_file_path):
    """
    Extract MFCC, Mel-Spectrogram, and Chroma features from an audio file.
    """
    # Load audio file
    y, sr = librosa.load(audio_file_path)

    # Extract features
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    mel_spectrogram = librosa.feature.melspectrogram(y=y, sr=sr)
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)

    return mfccs, mel_spectrogram, chroma


# Define the feature processing function
def process_features(features, target_size=(256, 256)):
    """
    Process each feature type to fit the target resolution, including normalization and resizing.
    """
    processed_features = []
    for feature in features:
        # Normalize features to a 0-1 range
        scaler = MinMaxScaler()
        feature_norm = scaler.fit_transform(feature)

        # Resize feature to target size
        zoom_factors = (target_size[0] / feature_norm.shape[0], target_size[1] / feature_norm.shape[1])
        feature_resized = zoom(feature_norm, zoom_factors, order=1)  # Linear interpolation

        processed_features.append(feature_resized)

    return processed_features


# Define the image creation function
def create_feature_image(features, output_path):
    """
    Combine processed features into a single RGB image and save it.
    """
    # Stack the features along the third dimension to form an RGB image
    rgb_image = np.stack(features, axis=-1)

    # Scale the values to the 0-255 range and convert to unsigned 8-bit integer
    rgb_image_scaled = np.uint8(rgb_image * 255)

    # Convert the numpy array to an image
    img = Image.fromarray(rgb_image_scaled, 'RGB')

    # Save the image
    img.save(output_path)


# Define the main function to create the dataset
def create_rgb_feature_dataset(audio_dir, output_dir, emotions):
    """
    Iterate over audio files, extract features, process them, generate RGB images, and save in a structured dataset.
    """
    for emotion in emotions:
        emotion_audio_dir = os.path.join(audio_dir, emotion)
        emotion_output_dir = os.path.join(output_dir, emotion)

        # Create the output directory if it doesn't exist
        os.makedirs(emotion_output_dir, exist_ok=True)

        for filename in os.listdir(emotion_audio_dir):
            if filename.endswith(".wav"):
                audio_file_path = os.path.join(emotion_audio_dir, filename)
                output_path = os.path.join(emotion_output_dir, filename.replace(".wav", ".png"))

                # Extract audio features
                mfccs, mel_spectrogram, chroma = extract_audio_features(audio_file_path)

                # Process features
                processed_features = process_features([mfccs, mel_spectrogram, chroma])

                # Create and save the feature image
                create_feature_image(processed_features, output_path)


# Commenting out the function call for development stage
create_rgb_feature_dataset(DATASET_AUDIO, OUTPUT_IMAGES, EMOTIONS)
