from preprocessing_SAVEE import load_data

import numpy as np
from model import build_transformer_model
from keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Load preprocessed data
X_train, X_test, y_train, y_test = load_data("C:/Users/Pana/Desktop/Northumbria/Final Year/Individual Computing "
                                             "Project KV6003BNN01/datasets/SAVEE/")

# Assuming X_train.shape[1] gives the feature vector length
input_shape = X_train.shape[1]
num_classes = len(np.unique(y_train))

# Initialize and compile the Transformer model
model = build_transformer_model(input_shape, num_classes)

# Callbacks
early_stopping = EarlyStopping(monitor='val_loss', patience=5, verbose=1)
model_checkpoint = ModelCheckpoint('best_transformer_model.h5', save_best_only=True, monitor='val_loss', mode='min')

# Training
model.fit(X_train, y_train, epochs=20, batch_size=32, validation_split=0.2,
          callbacks=[early_stopping, model_checkpoint])

# Evaluation
test_loss, test_accuracy = model.evaluate(X_test, y_test)
print(f"Test Loss: {test_loss}, Test Accuracy: {test_accuracy}")

# Make predictions
y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)

# Generate and print classification report
print("Classification Report:")
print(classification_report(y_test, y_pred_classes))

# Plot confusion matrix
cm = confusion_matrix(y_test, y_pred_classes)
plt.figure(figsize=(10, 7))
sns.heatmap(cm, annot=True, fmt="d", cmap='Blues')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix')
plt.show()
