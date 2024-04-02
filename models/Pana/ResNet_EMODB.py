# Import necessary libraries
import os
import numpy as np
from keras_preprocessing.image import img_to_array, load_img
from keras.applications.resnet import ResNet50, preprocess_input
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
from keras.optimizers import Adam
from sklearn.metrics import classification_report
from keras.callbacks import EarlyStopping

# Constants
DATA_DIR = "C:/Users/Pana/Desktop/Northumbria/Final Year/Individual Computing Project KV6003BNN01/datasets/Mixed/EMODB/"
IMAGE_SUBFOLDER = 'images'
EMOTIONS = ['anger', 'boredom', 'disgust', 'fear', 'happiness', 'neutral', 'sadness']
NUM_CLASSES = len(EMOTIONS)
IMAGE_SIZE = (256, 256)  # Adjust based on your images' dimensions
BATCH_SIZE = 32
EPOCHS = 100


# Function to load and preprocess images
def load_and_preprocess_image(img_path):
    img = load_img(img_path, target_size=IMAGE_SIZE)
    img = img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = preprocess_input(img)
    return img


# Initialize ResNet50 model for classification
def build_model(num_classes):
    base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(*IMAGE_SIZE, 3))
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(1024, activation='relu')(x)
    predictions = Dense(num_classes, activation='softmax')(x)
    model = Model(inputs=base_model.input, outputs=predictions)

    # Compile the model
    model.compile(optimizer=Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])
    return model


# Load dataset
def load_dataset(data_dir):
    features = []
    labels = []

    for emotion in EMOTIONS:
        print(f"Processing {emotion} images...")
        emotion_image_path = os.path.join(data_dir, IMAGE_SUBFOLDER, emotion)
        for image_file in os.listdir(emotion_image_path):
            full_image_path = os.path.join(emotion_image_path, image_file)
            img = load_and_preprocess_image(full_image_path)
            features.append(img)
            labels.append(EMOTIONS.index(emotion))

    features = np.vstack(features)
    labels = to_categorical(labels, num_classes=NUM_CLASSES)

    return features, labels


# Split the dataset
def split_dataset(features, labels, test_size=0.2):
    return train_test_split(features, labels, test_size=test_size, random_state=42)


# Main function to execute the process
def main(data_dir):
    model = build_model(NUM_CLASSES)
    features, labels = load_dataset(data_dir)
    X_train, X_test, y_train, y_test = split_dataset(features, labels)

    # Early Stopping
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, verbose=1, restore_best_weights=True)

    # Train the model
    model.fit(X_train, y_train, batch_size=BATCH_SIZE, epochs=EPOCHS, validation_data=(X_test, y_test), callbacks=early_stopping)

    # Predictions for the test set
    y_pred = model.predict(X_test)
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_true_classes = np.argmax(y_test, axis=1)

    # Classification report
    print(classification_report(y_true_classes, y_pred_classes, target_names=EMOTIONS))

    # Save the model for later use
    # model.save('emotion_recognition_model.h5')


# Note: Ensure you update DATA_DIR with the actual path to your dataset before running this function
# main(DATA_DIR)

print("Code ready with classification report added at the end of main().")

if __name__ == "__main__":
    main(DATA_DIR)
