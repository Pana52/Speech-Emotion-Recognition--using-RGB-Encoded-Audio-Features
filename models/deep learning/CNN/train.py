import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from preprocessing_CREMAD import load_data
from model import create_model
from keras.utils import to_categorical
from sklearn.metrics import classification_report, confusion_matrix
from keras.models import load_model
from keras.callbacks import EarlyStopping

# Path to your dataset
data_path = 'C:/Users/Pana/Desktop/Northumbria/Final Year/Individual Computing Project KV6003BNN01/datasets/CREMAD/'

# Load the data
X_train, X_test, y_train, y_test = load_data(data_path)

# One-hot encode labels
num_classes = len(np.unique(y_train))
y_train_encoded = to_categorical(y_train, num_classes)
y_test_encoded = to_categorical(y_test, num_classes)

# Assuming that all our features have the same shape
input_shape = (X_train.shape[1], 1)  # CNNs require a 3D input shape (batch_size, steps, input_dim)

# Reshape the training data to fit the model input shape
X_train = np.expand_dims(X_train, axis=2)
X_test = np.expand_dims(X_test, axis=2)

# Create the model
model = create_model(input_shape=input_shape, num_classes=num_classes)

# Setup Early Stopping
early_stopping = EarlyStopping(monitor='val_loss', patience=10, verbose=1, mode='min', restore_best_weights=True)

# Train the model with Early Stopping
epochs = 50  # You can adjust this
batch_size = 32  # You can adjust this
history = model.fit(X_train, y_train_encoded, epochs=epochs, batch_size=batch_size, validation_data=(X_test, y_test_encoded), callbacks=[early_stopping])

# Save the model
model_path = 'emotion_recognition_model.h5'
model.save(model_path)

# Load the model
model = load_model(model_path)

# Evaluate the model
test_loss, test_acc = model.evaluate(X_test, y_test_encoded, verbose=2)
print(f'Test accuracy: {test_acc}, Test loss: {test_loss}')

# Plot training history
plt.figure(figsize=(8, 8))
plt.subplot(2, 1, 1)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.legend()
plt.title('Accuracy')

plt.subplot(2, 1, 2)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.legend()
plt.title('Loss')
plt.show()

# Confusion Matrix and Classification Report
y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true_classes = np.argmax(y_test_encoded, axis=1)  # Convert one-hot to class numbers for comparison

print('Classification Report')
print(classification_report(y_true_classes, y_pred_classes))

cm = confusion_matrix(y_true_classes, y_pred_classes)
plt.figure(figsize=(10, 7))
sns.heatmap(cm, annot=True, fmt='d')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()

