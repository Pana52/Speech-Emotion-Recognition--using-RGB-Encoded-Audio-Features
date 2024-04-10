# Let's outline the functions and structure needed for the process
import os
from PIL import Image
import numpy as np
import librosa
from sklearn.preprocessing import MinMaxScaler
from scipy.interpolate import interp2d

DATASET_AUDIO = "/datasets/Mixed/EMODB/Audio/"
OUTPUT_IMAGES = "/datasets/Mixed/EMODB/3CF_512/"
EMOTIONS = ['anger', 'boredom', 'disgust', 'fear', 'happiness', 'neutral', 'sadness']


def interpolate_feature(feature, target_size=(512, 512), interp_kind='cubic'):
    # Generate x and y indices for the original feature size
    x_old = np.linspace(0, 1, feature.shape[1])
    y_old = np.linspace(0, 1, feature.shape[0])

    # Generate new indices for the interpolated array
    x_new = np.linspace(0, 1, target_size[1])
    y_new = np.linspace(0, 1, target_size[0])

    # Create 2D interpolation function
    interp_func = interp2d(x_old, y_old, feature, kind=interp_kind)

    # Interpolate the feature to the new resolution
    feature_interp = interp_func(x_new, y_new)

    return feature_interp


def extract_audio_features(audio_file_path, sr=48000, n_fft=4096, hop_length=1024, n_mels=256):
    """
    Extract Mel-Spectrogram, Spectral Centroid, and Zero Crossing Rate features from an audio file
    with parameters adjusted to increase feature resolution.
    """
    y, sr = librosa.load(audio_file_path, sr=sr)
    mel_spectrogram = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels)
    spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr, n_fft=n_fft, hop_length=hop_length)
    zero_crossing_rate = librosa.feature.zero_crossing_rate(y, hop_length=hop_length)
    return mel_spectrogram, spectral_centroid, zero_crossing_rate


def process_features(features, target_size=(512, 512)):
    """
    Process each feature type to fit the target resolution, including normalization and advanced resizing using cubic interpolation.
    """
    processed_features = []

    for feature in features:
        # If the feature is 1D (as spectral centroid and ZCR typically are), we expand it to 2D.
        if feature.ndim == 1:
            feature = feature.reshape(1, -1)
        elif feature.ndim == 2 and feature.shape[0] == 1:  # if shape is (1, N)
            feature = feature.T

        # Perform 2D interpolation to the desired size
        feature_interp = interpolate_feature(feature, target_size, interp_kind='cubic')

        # Normalize the interpolated feature to be in the range [0, 1]
        scaler = MinMaxScaler()
        feature_normalized = scaler.fit_transform(feature_interp)

        processed_features.append(feature_normalized)

    return processed_features


def create_feature_image(features, output_path):
    """
    Combine processed features into a single RGB image and save it.
    Apply rotations to the second and third channels.
    """
    # Apply rotations
    # features[1] = rotate(features[1], angle=0, reshape=False)
    # features[2] = rotate(features[2], angle=0, reshape=False)

    rgb_image = np.stack(features, axis=-1)
    rgb_image_scaled = np.uint8(rgb_image * 255)
    img = Image.fromarray(rgb_image_scaled, 'RGB')
    img.save(output_path)


def create_rgb_feature_dataset(audio_dir, output_dir, emotions):
    """
    Iterate over audio files, extract features, process them, generate RGB images, and save in a structured dataset.
    """
    for emotion in emotions:
        emotion_audio_dir = os.path.join(audio_dir, emotion)
        emotion_output_dir = os.path.join(output_dir, emotion)
        os.makedirs(emotion_output_dir, exist_ok=True)

        for filename in os.listdir(emotion_audio_dir):
            if filename.endswith(".wav"):
                audio_file_path = os.path.join(emotion_audio_dir, filename)
                output_path = os.path.join(emotion_output_dir, filename.replace(".wav", ".png"))
                mfccs, mel_spectrogram, chroma = extract_audio_features(audio_file_path)
                processed_features = process_features([mfccs, mel_spectrogram, chroma])
                create_feature_image(processed_features, output_path)


# Comment out the function call for operational security within the Python Code Interpreter environment.
create_rgb_feature_dataset(DATASET_AUDIO, OUTPUT_IMAGES, EMOTIONS)