# Import necessary libraries
import os
import numpy as np
from keras import Input
from keras_preprocessing.image import img_to_array, load_img
from keras.applications.resnet import ResNet50, preprocess_input
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
from keras.optimizers import Adam
from sklearn.metrics import classification_report
from keras.callbacks import EarlyStopping
from sklearn.cluster import KMeans

# Constants
DATA_DIR = "C:/Users/Pana/Desktop/Northumbria/Final Year/Individual Computing Project KV6003BNN01/datasets/Mixed/EMODB/"
IMAGE_SUBFOLDER = '4Channels'
EMOTIONS = ['anger', 'boredom', 'disgust', 'fear', 'happiness', 'neutral', 'sadness']
NUM_CLASSES = len(EMOTIONS)
IMAGE_SIZE = (256, 256)
BATCH_SIZE = 32
EPOCHS = 500
PATIENCE = 20
LEARNING_RATE = 0.0001


def load_and_preprocess_image(img_path, channel_idx):
    img = load_img(img_path, color_mode='rgba', target_size=IMAGE_SIZE)
    img = img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = preprocess_input(img)
    # Select a single channel
    return img[:, :, :, channel_idx:channel_idx + 1]


def load_dataset_for_channel(data_dir, channel_idx):
    features = []
    labels = []
    for emotion in EMOTIONS:
        print(f"Processing {emotion} images for channel {channel_idx}...")
        emotion_image_path = os.path.join(data_dir, IMAGE_SUBFOLDER, emotion)
        for image_file in os.listdir(emotion_image_path):
            full_image_path = os.path.join(emotion_image_path, image_file)
            img = load_and_preprocess_image(full_image_path, channel_idx)
            features.append(np.squeeze(img))
            labels.append(EMOTIONS.index(emotion))
    features = np.array(features)
    labels = to_categorical(labels, num_classes=NUM_CLASSES)
    return features, labels


def build_model(input_shape=(256, 256, 1), use_pretrained_weights=False):
    if use_pretrained_weights:
        # Initialize with pre-trained weights and modify later (for 3-channel inputs)
        base_model = ResNet50(weights='imagenet', include_top=False, input_tensor=Input(shape=(256, 256, 3)))
        # Custom logic to adjust the first convolutional layer weights for 1-channel input
        # This part is non-trivial and requires careful manipulation of the weights
    else:
        # Initialize without pre-trained weights for 1-channel input
        base_model = ResNet50(weights=None, include_top=False, input_tensor=Input(shape=input_shape))

    x = GlobalAveragePooling2D()(base_model.output)
    x = Dense(1024, activation='relu')(x)
    predictions = Dense(NUM_CLASSES, activation='softmax')(x)
    model = Model(inputs=base_model.input, outputs=predictions)
    model.compile(optimizer=Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])
    return model


def train_model_for_channel(data_dir, channel_idx):
    print(f"Training model for channel {channel_idx}")
    features, labels = load_dataset_for_channel(data_dir, channel_idx)
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

    model = build_model(input_shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 1))

    early_stopping = EarlyStopping(monitor='val_loss', patience=PATIENCE, verbose=1, restore_best_weights=True)
    model.fit(X_train, y_train, batch_size=BATCH_SIZE, epochs=EPOCHS, validation_data=(X_test, y_test),
              callbacks=[early_stopping])

    # Save the model
    # model.save(f'model_channel_{channel_idx}.h5')

    return model, X_test, y_test


def combine_predictions(models, X_tests):
    predictions = [model.predict(X_test) for model, X_test in zip(models, X_tests)]
    final_prediction = np.mean(predictions, axis=0)
    return final_prediction


def main(data_dir):
    models = []
    X_tests = []
    y_test = None
    for channel_idx in range(4):  # RGBA channels
        model, X_test, test_labels = train_model_for_channel(data_dir, channel_idx)
        models.append(model)
        X_tests.append(X_test)
        if y_test is None:
            y_test = test_labels

    final_prediction = combine_predictions(models, X_tests)
    y_pred_classes = np.argmax(final_prediction, axis=1)
    y_true_classes = np.argmax(y_test, axis=1)

    print(classification_report(y_true_classes, y_pred_classes, target_names=EMOTIONS))


if __name__ == "__main__":
    main(DATA_DIR)

