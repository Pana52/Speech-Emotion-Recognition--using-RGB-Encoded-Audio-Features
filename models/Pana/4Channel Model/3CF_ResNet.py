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
IMAGE_SUBFOLDER = '3CF_Images'
EMOTIONS = ['anger', 'boredom', 'disgust', 'fear', 'happiness', 'neutral', 'sadness']
NUM_CLASSES = len(EMOTIONS)
IMAGE_SIZE = (256, 256)  # Adjust based on your images' dimensions
BATCH_SIZE = 32
EPOCHS = 500
PATIENCE = 50
LEARNING_RATE = 0.0001
N_CLUSTERS = 10  # Number of clusters for K-means


# Function to load, preprocess images, and extract features
def load_and_extract_features(img_path, feature_model):
    img = load_img(img_path, target_size=IMAGE_SIZE)
    img = img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = preprocess_input(img)
    features = feature_model.predict(img)
    return np.squeeze(features)


# Function to build a feature extraction model based on ResNet50
def build_feature_extractor():
    base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(*IMAGE_SIZE, 3))
    model = Model(inputs=base_model.input, outputs=GlobalAveragePooling2D()(base_model.output))
    return model


# Function to apply clustering on the extracted features
def apply_clustering(features):
    kmeans = KMeans(n_clusters=N_CLUSTERS, random_state=42)
    cluster_labels = kmeans.fit_predict(features)
    return cluster_labels, kmeans.cluster_centers_


# Initialize ResNet50 model for classification
def build_classification_model(num_classes, input_shape=(2048,)):
    input_layer = Input(shape=input_shape)
    x = Dense(1024, activation='relu')(input_layer)
    predictions = Dense(num_classes, activation='softmax')(x)
    model = Model(inputs=input_layer, outputs=predictions)

    # Compile the model
    model.compile(optimizer=Adam(learning_rate=LEARNING_RATE), loss='categorical_crossentropy', metrics=['accuracy'])
    return model


# Load dataset and extract features
def load_dataset_and_extract_features(data_dir, feature_model):
    features = []
    labels = []

    for emotion in EMOTIONS:
        print(f"Processing {emotion} images...")
        emotion_image_path = os.path.join(data_dir, IMAGE_SUBFOLDER, emotion)
        for image_file in os.listdir(emotion_image_path):
            full_image_path = os.path.join(emotion_image_path, image_file)
            img_features = load_and_extract_features(full_image_path, feature_model)
            features.append(img_features)
            labels.append(EMOTIONS.index(emotion))

    features = np.array(features)
    labels = to_categorical(labels, num_classes=NUM_CLASSES)

    return features, labels


# Main function to execute the process
def main(data_dir):
    # Build and use the feature extractor
    feature_extractor = build_feature_extractor()
    features, labels = load_dataset_and_extract_features(data_dir, feature_extractor)

    # Apply clustering
    cluster_labels, _ = apply_clustering(features)

    # Consider clustering labels as additional feature or use them to transform your features
    # Here we directly use the labels. For a more sophisticated approach, consider using distances or other statistics.

    # Proceed with classification
    classification_model = build_classification_model(NUM_CLASSES, input_shape=(2048,))
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

    # Early Stopping
    early_stopping = EarlyStopping(monitor='val_loss', patience=PATIENCE, verbose=1, restore_best_weights=True)

    # Train the model
    classification_model.fit(X_train, y_train, batch_size=BATCH_SIZE, epochs=EPOCHS, validation_data=(X_test, y_test),
                             callbacks=[early_stopping])

    # Predictions for the test set
    y_pred = classification_model.predict(X_test)
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_true_classes = np.argmax(y_test, axis=1)

    # Classification report
    print(classification_report(y_true_classes, y_pred_classes, target_names=EMOTIONS))

    # Save the model for later use
    # classification_model.save('emotion_recognition_model.h5')


if __name__ == "__main__":
    main(DATA_DIR)

