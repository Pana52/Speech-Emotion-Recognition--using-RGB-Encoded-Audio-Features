# Let's outline the functions and structure needed for the process
import os
import random

from PIL import Image
import numpy as np
import librosa
from sklearn.preprocessing import MinMaxScaler
from scipy.ndimage import zoom

DATASET_AUDIO = "C:/Users/Pana/Desktop/Northumbria/Final Year/Individual Computing Project KV6003BNN01/datasets/audio/EMODB/"
OUTPUT_IMAGES = "C:/Users/Pana/Desktop/Northumbria/Final Year/Individual Computing Project KV6003BNN01/datasets/Mixed/EMODB/128p/3CF/3cf_random"
# EMODB
EMOTIONS = ['anger', 'boredom', 'disgust', 'fear', 'happiness', 'neutral', 'sadness']
# RAVDESS
# EMOTIONS = ['angry', 'calm', 'disgust', 'fearful', 'happy', 'neutral', 'sad', 'surprised']


# Feature extraction function remains the same
def extract_audio_features(audio_file_path):
    """
    Extract MFCC, Mel-Spectrogram, and Chroma features from an audio file.
    """
    # Load audio file
    y, sr = librosa.load(audio_file_path)

    # Extract features
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
    mel_spectrogram = librosa.feature.melspectrogram(y=y, sr=sr)
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)

    return mfccs, mel_spectrogram, chroma


# Updating the feature processing function for direct greyscale image preparation
def process_features(features, target_size=(128, 128)):
    """
    Process each feature type to fit the target resolution, including normalization and resizing.
    Each feature is treated as a greyscale image.
    """
    processed_features = []
    for feature in features:
        # Normalize features to a 0-1 range
        scaler = MinMaxScaler()
        feature_norm = scaler.fit_transform(feature)

        # Resize feature to target size, preserving the greyscale image properties
        zoom_factors = (target_size[0] / feature_norm.shape[0], target_size[1] / feature_norm.shape[1])
        feature_resized = zoom(feature_norm, zoom_factors, order=1)  # Linear interpolation

        processed_features.append(feature_resized)

    return processed_features


# Updating the image creation function to handle greyscale images as RGB channels
# Update the create_feature_image function to allow flexible feature channel ordering
def create_feature_image(features, output_path, channel_order=[0, 1, 2]):
    """
    Combine processed features into a single RGB image and save it, with customizable channel order.

    Parameters:
    - features: List of processed features.
    - output_path: Path to save the RGB image.
    - channel_order: List specifying the order of features for the RGB channels.
    """
    # Reorder the features based on the specified channel order
    reordered_features = [features[i] for i in channel_order]

    # Stack the reordered features along the third dimension to form an RGB image
    rgb_image = np.stack(reordered_features, axis=-1)

    # Ensure the RGB image is scaled to the 0-255 range and convert to unsigned 8-bit integer
    rgb_image_scaled = np.uint8(rgb_image * 255)

    # Convert the numpy array to an image
    img = Image.fromarray(rgb_image_scaled, 'RGB')

    # Save the image
    img.save(output_path)


# Update the main dataset creation function to include the feature order parameter
def create_rgb_feature_dataset(audio_dir, output_dir, emotions):
    for emotion in emotions:
        emotion_audio_dir = os.path.join(audio_dir, emotion)
        emotion_output_dir = os.path.join(output_dir, emotion)
        os.makedirs(emotion_output_dir, exist_ok=True)

        for filename in os.listdir(emotion_audio_dir):
            if filename.endswith(".wav"):
                audio_file_path = os.path.join(emotion_audio_dir, filename)
                output_path = os.path.join(emotion_output_dir, filename.replace(".wav", ".png"))

                # Extract audio features
                mfccs, mel_spectrogram, chroma = extract_audio_features(audio_file_path)

                # Process features
                processed_features = process_features([mfccs, mel_spectrogram, chroma])

                # Generate a random feature order
                feature_order = [0, 1, 2]
                random.shuffle(feature_order)

                # Create and save the feature image with the randomized feature order
                create_feature_image(processed_features, output_path, feature_order)


create_rgb_feature_dataset(DATASET_AUDIO, OUTPUT_IMAGES, EMOTIONS)
