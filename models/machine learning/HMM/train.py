# train_CREMAD.py - Extended with evaluation
from preprocessing_EMODB import load_data

from model import create_hmm_model
import numpy as np
from sklearn.metrics import accuracy_score, classification_report


def train_model(data_path):
    """
    Trains the HMM model on the preprocessed CREMA-D dataset and evaluates its performance.

    :param data_path: Path to the dataset directory.
    """
    # Load and preprocess the data
    X_train, X_test, y_train, y_test = load_data(data_path)

    # Initialize the HMM model
    hmm_model = create_hmm_model(n_components=len(np.unique(y_train)))

    # Training the model
    # Note: Ensure X_train is in a sequence format suitable for HMM if needed
    hmm_model.fit(X_train)

    # Predicting the test set results
    # Note: This step may need adjustment based on how your HMM model outputs predictions
    y_pred = hmm_model.predict(X_test)

    # Evaluating the model
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("\nClassification Report:\n", classification_report(y_test, y_pred))


if __name__ == "__main__":
    data_path = "C:/Users/Pana/Desktop/Northumbria/Final Year/Individual Computing Project " \
                "KV6003BNN01/datasets/EMODB/"
    train_model(data_path)
