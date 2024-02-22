from preprocessing import load_data
from model import create_model
import numpy as np


def main():
    data_path = "C:/Users/Pana/Desktop/Northumbria/Final Year/Individual Computing Project KV6003BNN01/datasets/CREMAD/"
    # Load and preprocess the data
    X_train, X_test, y_train, y_test = load_data(data_path)

    # Assuming features shape is (num_samples, num_features) for input_shape calculation
    input_shape = (X_train.shape[1],)
    num_classes = len(np.unique(y_train))

    # Create the MLP model
    model = create_model(input_shape, num_classes)

    # Train the model
    model.fit(X_train, y_train, epochs=30, batch_size=32, validation_data=(X_test, y_test))


if __name__ == '__main__':
    main()
