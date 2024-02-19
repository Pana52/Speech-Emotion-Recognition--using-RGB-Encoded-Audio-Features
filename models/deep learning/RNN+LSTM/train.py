from preprocessing import load_data, split_data
from model import create_model
import numpy as np

# Define the path to your dataset
data_dir = 'C:/Users/Pana/Desktop/Northumbria/Final Year/Individual Computing Project KV6003BNN01/datasets/CREMAD/'

# Load and preprocess the data
X, y = load_data(data_dir, augment=True)
X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y)

# Assuming all sequences are padded to the same length, we can infer the input shape
input_shape = X_train.shape[1:]  # Shape of MFCCs, excluding batch size
num_classes = y_train.shape[1]  # Number of emotion categories

# Create the model
model = create_model(input_shape, num_classes)

# Train the model
model.fit(X_train, y_train, epochs=30, batch_size=64, validation_data=(X_val, y_val))

# Evaluate the model on the test set
test_loss, test_acc = model.evaluate(X_test, y_test, verbose=2)
print(f'Test accuracy: {test_acc}, Test loss: {test_loss}')
