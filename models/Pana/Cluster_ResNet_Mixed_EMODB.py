import os
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from keras.models import Model
from keras.layers import Input, Dense, Flatten, concatenate, GlobalAveragePooling2D
from keras.applications.resnet import ResNet50
from keras.preprocessing.image import ImageDataGenerator

# Variables
dataset_path = "C:/Users/Pana/Desktop/Northumbria/Final Year/Individual Computing Project " \
               "KV6003BNN01/datasets/Mixed/EMODB/audio/"
image_output_dir = 'C:/Users/Pana/Desktop/Northumbria/Final Year/Individual Computing Project ' \
                   'KV6003BNN01/datasets/Mixed/EMODB/Images/'
EMOTIONS = ['anger', 'boredom', 'disgust', 'fear', 'happiness', 'neutral', 'sadness']
EPOCH = 100
BATCH_SIZE = 32
SAMPLE_RATE = 48000
N_FFT = 2048
HOP_L = 512
N_MELS = 128


def extract_mel_spectrogram(audio, sample_rate, n_fft=N_FFT, hop_length=HOP_L, n_mels=N_MELS):
    S = librosa.feature.melspectrogram(y=audio, sr=sample_rate, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels)
    S_DB = librosa.power_to_db(S, ref=np.max)
    return S_DB


def mel_spectrogram_to_image(S_DB, filename):
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(S_DB, sr=SAMPLE_RATE, x_axis='time', y_axis='mel')
    plt.colorbar(format='%+2.0f dB')
    plt.title('Mel-spectrogram')
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()


# Adjust this function based on the new approach
def extract_features_and_save_images(audio, sample_rate, image_save_path, emotion):
    emotion_dir = os.path.join(image_output_dir, emotion)
    if not os.path.exists(emotion_dir):
        os.makedirs(emotion_dir)
    filename_with_path = os.path.join(emotion_dir, os.path.basename(image_save_path))
    S_DB = extract_mel_spectrogram(audio, sample_rate)
    mel_spectrogram_to_image(S_DB, filename_with_path)
    return S_DB  # Return mel-spectrogram features for further processing


# Data Loading and Preprocessing with adjustments
def load_and_preprocess_data_with_dual_inputs(dataset_path, image_output_dir):
    vector_features, labels, image_filenames = [], [], []
    if not os.path.exists(image_output_dir):
        os.makedirs(image_output_dir)
    for label, emotion in enumerate(EMOTIONS):
        emotion_path = os.path.join(dataset_path, emotion)
        for filename in os.listdir(emotion_path):
            file_path = os.path.join(emotion_path, filename)
            audio, sample_rate = librosa.load(file_path, sr=SAMPLE_RATE)
            image_filename = f"{filename.split('.')[0]}.png"
            full_image_path = os.path.join(image_output_dir, emotion, image_filename)
            features = extract_features_and_save_images(audio, sample_rate, full_image_path, emotion)
            vector_features.append(features)
            labels.append(label)
            image_filenames.append(full_image_path)
    reduced_vector_features = reduce_vector_features(vector_features)  # Reduce feature dimensions
    return reduced_vector_features, np.array(labels, dtype=int), image_filenames


# Simplified for illustration
def reduce_vector_features(vector_features):
    # Assuming vector_features is a list of 2D arrays (mel-spectrograms)
    reduced_features = []
    for feature in vector_features:
        # Average across the time dimension (axis=1)
        reduced_feature = np.mean(feature, axis=1)
        reduced_features.append(reduced_feature)
    return np.array(reduced_features)


# Model Definition with GlobalAveragePooling2D
def define_dual_input_model():
    image_input = Input(shape=(256, 256, 3))
    base_model = ResNet50(include_top=False, weights='imagenet', input_tensor=image_input)
    base_model.trainable = False
    image_branch = GlobalAveragePooling2D()(base_model.output)
    vector_input = Input(shape=(128,))
    vector_branch = Dense(256, activation='relu')(vector_input)
    vector_branch = Dense(128, activation='relu')(vector_branch)
    merged = concatenate([image_branch, vector_branch])
    merged_output = Dense(256, activation='relu')(merged)
    final_output = Dense(len(EMOTIONS), activation='softmax')(merged_output)
    model = Model(inputs=[image_input, vector_input], outputs=final_output)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model


def create_generators(batch_size, X_image_paths, X_vector_features, Y_labels):
    image_data_gen = ImageDataGenerator(rescale=1. / 255)

    # Assuming the images are already ordered according to Y_labels
    image_generator = image_data_gen.flow_from_directory(
        directory=image_output_dir,  # Make sure this is your correct image directory
        target_size=(256, 256),
        color_mode='rgb',
        classes=EMOTIONS,
        batch_size=batch_size,
        shuffle=True,  # Shuffling should match across both generators
        seed=42  # Ensuring reproducibility
    )

    def vector_data_generator(X_vector_features, batch_size):
        num_samples = len(X_vector_features)
        while True:  # Loop forever so the generator never terminates
            for offset in range(0, num_samples, batch_size):
                end = offset + batch_size
                batch_features = X_vector_features[offset:end]
                yield np.array(batch_features)

    # Adjusted call to the vector data generator without Y_labels
    vector_generator = vector_data_generator(X_vector_features, batch_size)

    while True:
        image_data, image_labels = next(image_generator)
        vector_data = next(vector_generator)
        yield [image_data, vector_data], image_labels


# Assuming the rest of the code for data generators and training remains the same
# Load and preprocess data
vector_features, labels, image_filenames = load_and_preprocess_data_with_dual_inputs(dataset_path, image_output_dir)

# Split the dataset into training, validation, and testing sets
X_train_images, X_temp_images, X_train_vectors, X_temp_vectors, y_train, y_temp = train_test_split(
    image_filenames, vector_features, labels, test_size=0.3, random_state=42, stratify=labels
)
X_val_images, X_test_images, X_val_vectors, X_test_vectors, y_val, y_test = train_test_split(
    X_temp_images, X_temp_vectors, y_temp, test_size=0.5, random_state=42, stratify=y_temp
)

# Transform labels into categorical (one-hot encoding)
y_train_cat = to_categorical(y_train, num_classes=len(EMOTIONS))
y_val_cat = to_categorical(y_val, num_classes=len(EMOTIONS))
y_test_cat = to_categorical(y_test, num_classes=len(EMOTIONS))

# Model Training Setup
model = define_dual_input_model()

# Data Generators -- Ensure the generators are prepared to handle the now correctly shaped vector features
train_generator = create_generators(BATCH_SIZE, X_train_images, X_train_vectors, y_train_cat)
validation_generator = create_generators(BATCH_SIZE, X_val_images, X_val_vectors, y_val_cat)

steps_per_epoch = np.ceil(len(X_train_images) / BATCH_SIZE)
validation_steps = np.ceil(len(X_val_images) / BATCH_SIZE)

# Training the model
if __name__ == "__main__":
    history = model.fit(
        train_generator,
        steps_per_epoch=len(X_train_images) // BATCH_SIZE,
        epochs=EPOCH,
        validation_data=validation_generator,
        validation_steps=len(X_val_images) // BATCH_SIZE
    )
