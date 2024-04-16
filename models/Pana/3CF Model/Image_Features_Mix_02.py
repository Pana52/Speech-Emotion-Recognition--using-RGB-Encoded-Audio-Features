# Let's outline the functions and structure needed for the process
import os
from PIL import Image
import numpy as np
import librosa
from sklearn.preprocessing import MinMaxScaler
from scipy.ndimage import zoom

DATASET_AUDIO = "C:/Users/Pana/Desktop/Northumbria/Final Year/Individual Computing Project KV6003BNN01/datasets/audio/EMODB/"
OUTPUT_IMAGES = "C:/Users/Pana/Desktop/Northumbria/Final Year/Individual Computing Project KV6003BNN01/datasets/Mixed/EMODB/256p/6F"
# EMODB
EMOTIONS = ['anger', 'boredom', 'disgust', 'fear', 'happiness', 'neutral', 'sadness']


# RAVDESS
# EMOTIONS = ['angry', 'calm', 'disgust', 'fearful', 'happy', 'neutral', 'sad', 'surprised']


# Feature extraction function remains the same
def extract_audio_features(audio_file_path):
    # Load the audio file
    y, sr = librosa.load(audio_file_path, sr=None)

    # Set minimum `n_fft`
    min_n_fft = 256
    n_fft = max(min_n_fft, min(len(y), 2048))

    # Pad if necessary
    if len(y) < n_fft:
        y = np.pad(y, pad_width=(0, n_fft - len(y)), mode='reflect')

    # Explicitly use calculated `n_fft`
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20, n_fft=n_fft)
    mel_spectrogram = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=n_fft)
    chroma = librosa.feature.chroma_stft(y=y, sr=sr, n_fft=n_fft)
    spectral_contrast = librosa.feature.spectral_contrast(y=y, sr=sr, n_fft=n_fft)
    tonnetz = librosa.feature.tonnetz(y=y, sr=sr)
    zero_crossing_rate = librosa.feature.zero_crossing_rate(y, frame_length=n_fft)[0]

    return mfccs, mel_spectrogram, chroma, spectral_contrast, tonnetz, zero_crossing_rate




# Updating the feature processing function for direct greyscale image preparation
def process_features(features, target_size=(256, 256)):
    """
    Process each feature type to fit the target resolution, including normalization and resizing.
    Each feature is treated as a greyscale image.
    """
    processed_features = []
    for feature in features:
        # Normalize features to a 0-1 range
        scaler = MinMaxScaler()
        feature_norm = scaler.fit_transform(feature if feature.ndim > 1 else feature[:, np.newaxis])

        # Resize feature to target size, preserving the greyscale image properties
        zoom_factors = (target_size[0] / feature_norm.shape[0], target_size[1] / feature_norm.shape[1])
        feature_resized = zoom(feature_norm, zoom_factors, order=1)  # Linear interpolation

        processed_features.append(feature_resized)

    return processed_features


# Updating the image creation function to handle greyscale images as RGB channels
# Update the create_feature_image function to allow flexible feature channel ordering
def create_feature_image(features, output_path):
    """
    Combine processed features into a single RGB image by automatically assigning:
    - The first two features to the Red channel
    - The next two features to the Green channel
    - The last two features to the Blue channel
    """
    if len(features) != 6:
        raise ValueError("Exactly six features must be provided.")

    # Combine the features into RGB channels by averaging pairs
    r_channel = np.mean(features[0:2], axis=0)
    g_channel = np.mean(features[2:4], axis=0)
    b_channel = np.mean(features[4:6], axis=0)

    # Stack the averaged channels to form an RGB image
    rgb_image = np.stack([r_channel, g_channel, b_channel], axis=-1)

    # Ensure the RGB image is scaled to the 0-255 range and convert to unsigned 8-bit integer
    rgb_image_scaled = np.uint8(rgb_image * 255)

    # Convert the numpy array to an image
    img = Image.fromarray(rgb_image_scaled, 'RGB')

    # Save the image
    img.save(output_path)


# Update the main dataset creation function to include the feature order parameter
def create_rgb_feature_dataset(audio_dir, output_dir, emotions):
    """
    Iterate over audio files, extract features, process them, generate RGB images by automatically assigning features to channels, and save in a structured dataset.

    Parameters:
    - audio_dir: Directory containing the audio files, organized by emotion.
    - output_dir: Output directory for the RGB images, organized by emotion.
    - emotions: List of emotions to process.
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
                features = extract_audio_features(audio_file_path)

                # Process features
                processed_features = process_features(features)

                # Create and save the feature image
                create_feature_image(processed_features, output_path)


create_rgb_feature_dataset(DATASET_AUDIO, OUTPUT_IMAGES, EMOTIONS)
