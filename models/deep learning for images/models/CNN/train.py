import numpy as np
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
from preprocessing import load_images_from_folder, preprocess_data, split_dataset, apply_augmentations
from model import create_model

# Load and preprocess the dataset
data_path = 'C:/Users/Pana/Desktop/Northumbria/Final Year/Individual Computing Project ' \
            'KV6003BNN01/Speech-Emotion-Recognition---Audio-Dataset/models/deep learning for ' \
            'images/datasets/RAVDESS/MELSPEC_100x100/'
image_size = (100, 100)
num_classes = 8
batches = 32
epochs = 100  # Adjust as needed


images, labels = load_images_from_folder(data_path, image_size=image_size)

# Define your augmentations here
# augmentations = {
    # 'horizontal_flip': True,
    # 'rotation': 25,  # Rotate by up to 25 degrees
    # 'noise': 0.02,   # Add Gaussian noise
    # 'brightness': 0.2,  # Adjust brightness
    # 'shear': 5  # Shear by 5 degrees
# }

# Apply augmentations
# images_augmented = apply_augmentations(images, augmentations=augmentations)

# Continue with your preprocessing
# images_augmented, labels = preprocess_data(images_augmented, labels)
# X_train, X_val, X_test, y_train, y_val, y_test = split_dataset(images_augmented, labels)

images, labels = preprocess_data(images, labels)
X_train, X_val, X_test, y_train, y_val, y_test = split_dataset(images, labels)

# Adjust your model configuration if necessary to handle the new shape of features
model = create_model(input_shape=(100, 100, 3), num_classes=num_classes)
model.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])

# Model training
# early_stopping = EarlyStopping(monitor='val_loss', patience=50, restore_best_weights=True)
history = model.fit(X_train, y_train, batch_size=batches, epochs=epochs, validation_data=(X_val, y_val))

# Evaluate the model on the test set
test_loss, test_acc = model.evaluate(X_test, y_test)
print(f'Test Accuracy: {test_acc:.2f}')

# Generate a classification report
y_pred = model.predict(X_test, batch_size=32)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true = np.argmax(y_test, axis=1)

# CREMA-D
# print(classification_report(y_true, y_pred_classes, target_names=['ANG', 'DIS', 'FEA', 'HAP', 'NEU', 'SAD']))

# EMO-DB
# print(classification_report(y_true, y_pred_classes, target_names=['Anger', 'Boredom', 'Disgust', 'Fear', 'Happiness', 'Neutral', 'Sadness']))

# RAVDESS
print(classification_report(y_true, y_pred_classes, target_names=['Neutral', 'Calm', 'Happy', 'Sad', 'Angry', 'Fearful', 'Disgust', 'Surprised']))
