import numpy as np
from preprocessing import load_data
from model import create_gmm_models
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split


def train_and_evaluate(data_path):
    """
    Train GMM models for each class and evaluate on test data.

    Parameters:
    - data_path: The path to the dataset directory.
    """
    X, y = load_data(data_path)  # This should return features and labels without splitting
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)  # Encode labels to integers

    # Split the dataset into training and testing
    X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

    n_classes = len(np.unique(y_encoded))
    gmm_models = create_gmm_models(n_components=16, n_classes=n_classes)

    # Train a GMM for each class
    for class_index in range(n_classes):
        # Select training data for this class
        X_train_class = X_train[y_train == class_index]
        gmm_models[class_index].fit(X_train_class)

    # Evaluate models
    correct = 0
    for x, true_class in zip(X_test, y_test):
        # Compute the likelihood of each model generating the observed data
        likelihoods = [gmm_models[class_index].score_samples(x.reshape(1, -1)) for class_index in range(n_classes)]
        predicted_class = np.argmax(likelihoods)
        correct += (predicted_class == true_class)

    accuracy = correct / len(X_test)
    print(f"Accuracy: {accuracy * 100:.2f}%")


if __name__ == "__main__":
    dataset_path = "C:/Users/Pana/Desktop/Northumbria/Final Year/Individual Computing Project " \
                   "KV6003BNN01/datasets/CREMAD/"  # Adjust this path
    train_and_evaluate(dataset_path)
