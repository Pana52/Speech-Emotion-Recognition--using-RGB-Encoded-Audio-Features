import numpy as np
from preprocessing import load_data
from model import create_model
from keras.utils import to_categorical

# Path to your dataset
data_path = 'C:/Users/Pana/Desktop/Northumbria/Final Year/Individual Computing Project KV6003BNN01/datasets/CREMAD/'

# Load the data
X_train, X_test, y_train, y_test = load_data(data_path)

# One-hot encode labels
num_classes = len(np.unique(y_train))
y_train = to_categorical(y_train, num_classes)
y_test = to_categorical(y_test, num_classes)

# Assuming that all our features have the same shape
input_shape = (X_train.shape[1], 1)  # CNNs require a 3D input shape (batch_size, steps, input_dim)

# Reshape the training data to fit the model input shape
X_train = np.expand_dims(X_train, axis=2)
X_test = np.expand_dims(X_test, axis=2)

# Create the model
model = create_model(input_shape=input_shape, num_classes=num_classes)

# Train the model
epochs = 50  # You can adjust this
batch_size = 32  # You can adjust this
model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(X_test, y_test))

# Evaluate the model
score = model.evaluate(X_test, y_test, verbose=0)
print(f'Test loss: {score[0]} / Test accuracy: {score[1]}')

# Save the model
model.save('emotion_recognition_model.h5')
