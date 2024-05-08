from preprocessing_RAVDESS import load_data
from model import create_gmm_models
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import LabelEncoder
import numpy as np

# Assuming the dataset is located in a 'dataset' directory
data_path = "PATH"


def main():
    # Load the data
    X_train, X_test, y_train, y_test = load_data(data_path)

    # Create GMM models for each class
    n_classes = len(np.unique(y_train))  # Assumes y_train contains numeric class labels
    gmm_models = create_gmm_models(n_components=16, n_classes=n_classes)

    # Train each GMM on its class
    for class_index in range(n_classes):
        # Find the data points corresponding to the current class and train the GMM on them
        X_train_class = X_train[y_train == class_index]
        gmm_models[class_index].fit(X_train_class)

    # Evaluate the models on the test set
    predictions = []
    for x_test in X_test:
        # Score each test sample against each model and pick the class with the highest score
        scores = np.array([gmm.score(x_test.reshape(1, -1)) for class_index, gmm in gmm_models.items()])
        predicted_class = np.argmax(scores)
        predictions.append(predicted_class)

    # Convert numeric predictions back to original labels if needed
    le = LabelEncoder()
    le.fit(np.unique(y_train))  # Fit label encoder to unique classes
    y_test_labels = le.inverse_transform(y_test)  # Inverse transform to get original labels
    predicted_labels = le.inverse_transform(predictions)

    # Evaluate the model
    accuracy = accuracy_score(y_test, predictions)
    print(f'Model accuracy: {accuracy * 100:.2f}%')
    print("Classification Report:")
    print(classification_report(y_test_labels, predicted_labels))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test_labels, predicted_labels))


if __name__ == '__main__':
    main()
