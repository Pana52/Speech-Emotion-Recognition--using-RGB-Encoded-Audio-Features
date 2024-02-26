# train.py
from preprocessing_CREMAD import load_data
from model import KNNModel

# Assuming the CREMA-D dataset is located in 'data/CREMA-D' directory
data_path = "C:/Users/Pana/Desktop/Northumbria/Final Year/Individual Computing Project KV6003BNN01/datasets/CREMAD/"


def main():
    X_train, X_test, y_train, y_test = load_data(data_path)

    knn_model = KNNModel(n_neighbors=5)
    knn_model.train(X_train, y_train)

    accuracy = knn_model.evaluate(X_test, y_test)
    print(f'Model accuracy: {accuracy * 100:.2f}%')


if __name__ == '__main__':
    main()
