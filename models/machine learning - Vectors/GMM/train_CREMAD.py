import numpy as np
from preprocessing_CREMAD import load_data
from model import create_gmm_models
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# Load dataset
data_path = "C:/Users/Pana/Desktop/Northumbria/Final Year/Individual Computing Project KV6003BNN01/datasets/CREMAD/"
features, labels = load_data(data_path)

# Split dataset into training and testing set
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

# Create GMM models
n_classes = len(np.unique(y_train))
gmm_models = create_gmm_models(n_components=16, n_classes=n_classes)

# Train GMM models
for class_index, gmm in gmm_models.items():
    # Select only the data samples that belong to the current class
    X_train_class = X_train[y_train == class_index]
    if len(X_train_class) == 0:
        continue
    gmm.fit(X_train_class)

# Predict using GMM models
predictions = []
for x_test in X_test:
    scores = np.array([gmm.score(x_test.reshape(1, -1)) for gmm in gmm_models.values()])
    predicted_class = np.argmax(scores)
    predictions.append(predicted_class)

# Evaluate the model
print("Classification Report:")
print(classification_report(y_test, predictions, target_names=['ANG', 'DIS', 'FEA', 'HAP', 'NEU', 'SAD']))
