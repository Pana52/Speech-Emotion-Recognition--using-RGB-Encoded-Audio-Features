from preprocessing_CREMAD import load_data
from model import create_model
from sklearn.metrics import classification_report, accuracy_score


def train_and_evaluate(data_path):
    """
    Load the data, train the SVM model, and evaluate its performance.

    Parameters:
    - data_path: The path to the dataset directory.
    """
    # Load and preprocess the data
    X_train, X_test, y_train, y_test = load_data(data_path)

    # Create the SVM model
    model = create_model()

    # Train the model
    print("Training the model...")
    model.fit(X_train, y_train)

    # Predictions on the test set
    y_pred = model.predict(X_test)

    # Evaluate the model
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Model Accuracy: {accuracy * 100:.2f}%")
    print("Classification Report:")
    print(classification_report(y_test, y_pred))


if __name__ == "__main__":
    # Specify the path to your CREMA-D dataset
    dataset_path = "C:/Users/Pana/Desktop/Northumbria/Final Year/Individual Computing Project " \
                   "KV6003BNN01/datasets/CREMAD/"
    train_and_evaluate(dataset_path)
