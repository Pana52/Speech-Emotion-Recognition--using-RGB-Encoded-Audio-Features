from preprocessing_SAVEE import load_data
from model import create_model
from sklearn.metrics import classification_report
import joblib


def train_and_evaluate():
    # Load and preprocess data
    data_path = "PATH"
    X_train, X_test, y_train, y_test = load_data(data_path)

    # Create model
    model = create_model(n_estimators=100, random_state=42)

    # Train model
    model.fit(X_train, y_train)

    # Evaluate model
    predictions = model.predict(X_test)
    print(classification_report(y_test, predictions))

    # Save the trained model
    joblib.dump(model, 'emotion_classifier_rf.joblib')


if __name__ == "__main__":
    train_and_evaluate()
