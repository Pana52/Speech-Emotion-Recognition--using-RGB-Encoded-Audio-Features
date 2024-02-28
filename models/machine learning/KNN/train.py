from preprocessing_EMODB import load_data
from model import KNNModel
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# Assuming the CREMA-D dataset is located in 'data/CREMA-D' directory
data_path = "C:/Users/Pana/Desktop/Northumbria/Final Year/Individual Computing Project KV6003BNN01/datasets/EMODB/"


def main():
    X_train, X_test, y_train, y_test = load_data(data_path)

    knn_model = KNNModel(n_neighbors=5)
    knn_model.train(X_train, y_train)

    # Predictions on the test set
    y_pred = knn_model.predict(X_test)

    # Evaluation
    accuracy = accuracy_score(y_test, y_pred)
    print(f'Model accuracy: {accuracy * 100:.2f}%')

    # Detailed classification report
    print("Classification Report:")

    #CREMA-D
    # print(classification_report(y_test, y_pred, target_names=['ANG', 'DIS', 'FEA', 'HAP', 'NEU', 'SAD']))

    #RAVDESS
    print(classification_report(y_test, y_pred))
    # Confusion Matrix
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))


if __name__ == '__main__':
    main()
