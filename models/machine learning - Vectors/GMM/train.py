import numpy as np
from preprocessing_EMODB import load_data
from model import create_gmm_models
from sklearn.metrics import classification_report

# Path to the dataset
dataset_path = 'C:/Users/Pana/Desktop/Northumbria/Final Year/Individual Computing Project KV6003BNN01/datasets/EMODB/'

# Load the dataset
X_train, X_test, y_train, y_test = load_data(dataset_path)

# Create Gaussian Mixture Models for each class
n_classes = 7  # Assuming 7 emotions as defined in preprocessing_SAVEE.py
gmm_models = create_gmm_models(n_components=16, n_classes=n_classes)

# Train each GMM model with the data corresponding to its class
for class_index, gmm in gmm_models.items():
    X_train_class = X_train[y_train == class_index]
    if len(X_train_class) == 0:
        continue
    print(f"Training GMM for class {class_index} with {X_train_class.shape[0]} samples.")
    gmm.fit(X_train_class)

# Predict the class for each sample in the test set
predictions = []
for X_test_instance in X_test:
    log_likelihood = np.array([gmm.score(X_test_instance.reshape(1, -1)) for class_index, gmm in gmm_models.items()])
    predicted_class = np.argmax(log_likelihood)
    predictions.append(predicted_class)

# Evaluate the performance
print("Classification Report:")
print(classification_report(y_test, predictions))
