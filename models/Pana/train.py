from audio_preprocessing import load_audio_files, audio_to_melspectrogram
from image_preprocessing import load_and_preprocess_images
from cnn_feature_extractor import build_cnn_model, extract_features
from gbm_classifier import train_gbm_model
import numpy as np
from skimage.transform import resize


# Function to adjust Mel-Spectrograms to match the CNN input shape
def adjust_mel_spectrograms(audio_features, target_shape):
    adjusted_features = []
    for feature in audio_features:
        # Assuming feature is a 2D Mel-Spectrogram, resize and adjust channels
        resized_feature = resize(feature, (target_shape[0], target_shape[1]), anti_aliasing=True)

        # If CNN expects multiple channels, replicate the Mel-Spectrogram across channels
        if target_shape[2] == 3:
            resized_feature = np.stack((resized_feature,) * 3, axis=-1)

        adjusted_features.append(resized_feature)

    return np.array(adjusted_features)


def main():
    dataset_directory_audio = "C:/Users/Pana/Desktop/Northumbria/Final Year/Individual Computing Project " \
                              "KV6003BNN01/datasets/Mixed/EMODB/audio/"
    dataset_directory_images = "C:/Users/Pana/Desktop/Northumbria/Final Year/Individual Computing Project " \
                               "KV6003BNN01/datasets/Mixed/EMODB/images/"

    # Load and preprocess audio data
    audio_files = load_audio_files(dataset_directory_audio)
    audio_mel_spectrograms = [audio_to_melspectrogram(data, sr) for _, (data, sr) in audio_files.items()]

    # Load and preprocess image data
    images, image_labels = load_and_preprocess_images(dataset_directory_images)

    # Define the CNN input shape based on your processed images or requirements
    cnn_input_shape = (224, 224, 3)  # Example shape, adjust based on your data

    # Adjust the audio-derived Mel-Spectrograms to match the CNN input shape
    audio_features_adjusted = adjust_mel_spectrograms(audio_mel_spectrograms, cnn_input_shape)

    # Assuming labels for audio and image data are aligned and of the same length
    combined_labels = np.concatenate((image_labels, image_labels), axis=0)

    # Combine audio and image data for feature extraction
    combined_data = np.concatenate((audio_features_adjusted, images), axis=0)

    # Initialize and train the CNN model for feature extraction
    cnn_model = build_cnn_model(input_shape=cnn_input_shape, num_classes=len(np.unique(combined_labels)))
    # Note: Include your CNN model training logic here, e.g., cnn_model.fit(...)

    # Extract features from the combined dataset using the trained CNN
    features = extract_features(cnn_model, combined_data)

    # Train and evaluate the GBM classifier with the extracted features
    gbm_model, eval_result = train_gbm_model(features, combined_labels)


if __name__ == "__main__":
    main()
