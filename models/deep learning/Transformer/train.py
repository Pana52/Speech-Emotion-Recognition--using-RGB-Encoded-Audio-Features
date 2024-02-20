import numpy as np
from preprocessing import load_data
from model import build_transformer_model

# Load preprocessed data
X_train, X_test, y_train, y_test = load_data("C:/Users/Pana/Desktop/Northumbria/Final Year/Individual Computing "
                                             "Project KV6003BNN01/datasets/CREMAD/")

# Assuming X_train.shape[1] gives the feature vector length
input_shape = X_train.shape[1]
num_classes = len(np.unique(y_train))

# Initialize and compile the Transformer model
model = build_transformer_model(input_shape, num_classes)

# Training
model.fit(X_train, y_train, epochs=20, batch_size=32, validation_split=0.2)

# Evaluation
test_loss, test_accuracy = model.evaluate(X_test, y_test)
print(f"Test Loss: {test_loss}, Test Accuracy: {test_accuracy}")
