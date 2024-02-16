from preprocessing import get_data
from model import create_model
from keras.callbacks import EarlyStopping
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

data_path = 'C:/Users/Pana/Desktop/Northumbria/Final Year/Individual Computing Project KV6003BNN01/datasets/CREMAD/'
# Load and preprocess the data
X_train, X_test, y_train, y_test = get_data(data_path)

# Assuming the shape of your features is (None, 40) and you have 6 classes
model = create_model((X_train.shape[1], 1), 6)

# Reshape features for Conv1D
X_train = np.expand_dims(X_train, -1)
X_test = np.expand_dims(X_test, -1)

# Early Stopping
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

# Train the model
history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=100, batch_size=32, callbacks=[early_stopping])

# Evaluate the model
loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
print(f'Test Loss: {loss:.4f}, Test Accuracy: {accuracy:.4f}')

# Predictions for metrics and confusion matrix
y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true_classes = np.argmax(y_test, axis=1)

# Classification report
print(classification_report(y_true_classes, y_pred_classes, target_names=['ANG', 'DIS', 'FEA', 'HAP', 'NEU', 'SAD']))

# Confusion matrix
cm = confusion_matrix(y_true_classes, y_pred_classes)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['ANG', 'DIS', 'FEA', 'HAP', 'NEU', 'SAD'],
                                                    yticklabels=['ANG', 'DIS', 'FEA', 'HAP', 'NEU', 'SAD'])
plt.xlabel('Predicted labels')
plt.ylabel('True labels')
plt.title('Confusion Matrix')
plt.show()
