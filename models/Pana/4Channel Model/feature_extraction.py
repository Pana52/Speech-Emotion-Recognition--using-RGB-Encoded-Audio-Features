# Let's outline the functions and structure needed for the process
import os
from PIL import Image
import numpy as np
import librosa
from sklearn.preprocessing import MinMaxScaler

DATASET_AUDIO = "C:/Users/Pana/Desktop/Northumbria/Final Year/Individual Computing Project KV6003BNN01/datasets/Mixed/EMODB/Audio/"
DATASET_IMAGES = "C:/Users/Pana/Desktop/Northumbria/Final Year/Individual Computing Project KV6003BNN01/datasets/Mixed/EMODB/Images/"
DATASET_4CHANNEL = "C:/Users/Pana/Desktop/Northumbria/Final Year/Individual Computing Project KV6003BNN01/datasets/Mixed/EMODB/4channel/"
EMOTIONS = ['anger', 'boredom', 'disgust', 'fear', 'happiness', 'neutral', 'sadness']


def extract_audio_features(audio_file_path):
    """
    Extract relevant audio features from an audio file.
    """
    # Load audio file
    y, sr = librosa.load(audio_file_path)

    # Example feature extraction: MFCCs
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)

    # Additional features can be added here

    # Return the MFCCs as an example feature set
    return mfccs


def process_features(features, target_size=(256, 256)):
    """
    Process and reshape the features to fit the target resolution.
    """
    # Scale features to [0, 1] range
    scaler = MinMaxScaler()
    features_scaled = scaler.fit_transform(features)

    # Resize feature matrix to target_size
    # For simplicity, we'll flatten and then interpolate. Adjustments might be needed.
    features_flat = features_scaled.flatten()
    features_interp = np.interp(np.linspace(0, len(features_flat), num=target_size[0] * target_size[1]),
                                range(len(features_flat)), features_flat)
    features_reshaped = features_interp.reshape(target_size)

    return features_reshaped


def add_features_to_image(mel_spectrogram_path, features, output_path):
    """
    Combine processed features with an RGB Mel-Spectrogram image to form an RGBA image and save it.
    """
    # Load the Mel-Spectrogram image
    img = Image.open(mel_spectrogram_path).convert("RGB")

    # Convert the processed features into an Image object (as alpha channel)
    alpha_channel = Image.fromarray(np.uint8(features * 255), 'L')

    # Combine the RGB image with the new alpha channel
    rgba_image = Image.merge("RGBA", img.split() + (alpha_channel,))

    # Save the resulting image
    rgba_image.save(output_path, "PNG")


def create_rgba_dataset_with_classes(audio_dir, image_dir, output_dir, emotions):
    """
    Iterate over the audio files and corresponding Mel-Spectrogram images within each class subdirectory,
    process them, and save the new RGBA images, preserving the class structure.
    """
    for emotion in emotions:
        # Paths to the emotion-specific subdirectories in the audio and image directories
        emotion_audio_dir = os.path.join(audio_dir, emotion)
        emotion_image_dir = os.path.join(image_dir, emotion)
        emotion_output_dir = os.path.join(output_dir, emotion)

        # Create the output directory if it doesn't exist
        os.makedirs(emotion_output_dir, exist_ok=True)

        for filename in os.listdir(emotion_audio_dir):
            if filename.endswith(".wav"):
                audio_file_path = os.path.join(emotion_audio_dir, filename)
                mel_spectrogram_path = os.path.join(emotion_image_dir, filename.replace(".wav", ".png"))
                output_path = os.path.join(emotion_output_dir, filename.replace(".wav", "_rgba.png"))

                # Extract and process audio features
                features = extract_audio_features(audio_file_path)
                processed_features = process_features(features)

                # Add features to the Mel-Spectrogram image and save
                add_features_to_image(mel_spectrogram_path, processed_features, output_path)


# Note: Function calls are commented out for this development stage.
create_rgba_dataset_with_classes(DATASET_AUDIO, DATASET_IMAGES, DATASET_4CHANNEL, EMOTIONS)
