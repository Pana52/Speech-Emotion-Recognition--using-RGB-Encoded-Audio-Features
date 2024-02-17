import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
from model import AudioClassifier  # Ensure this is your model file
from preprocessing import load_data, split_dataset, normalize_features
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, multilabel_confusion_matrix

# Define the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load and preprocess data
data_path = 'C:/Users/Pana/Desktop/Northumbria/Final Year/Individual Computing Project KV6003BNN01/datasets/CREMAD/'
features, labels = load_data(data_path)
features_normalized = normalize_features(features)
X_train, X_test, y_train, y_test = split_dataset(features_normalized, labels)

# Convert labels to one-hot encoded format
num_classes = len(np.unique(y_train))
y_train_onehot = F.one_hot(torch.tensor(y_train), num_classes=num_classes).float()
y_test_onehot = F.one_hot(torch.tensor(y_test), num_classes=num_classes).float()

# Convert to PyTorch tensors and create DataLoader for batch processing
train_dataset = TensorDataset(torch.tensor(X_train).float(), y_train_onehot)
test_dataset = TensorDataset(torch.tensor(X_test).float(), y_test_onehot)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Model setup
num_features = X_train.shape[1]  # Number of features
model = AudioClassifier(num_features, num_classes)
model = model.to(device)

# Loss and optimizer
criterion = nn.BCEWithLogitsLoss()  # Binary Cross-Entropy loss for multi-label classification
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training
num_epochs = 25
for epoch in range(num_epochs):
    model.train()
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
    print(f'Epoch {epoch+1}/{num_epochs}, Loss: {loss.item():.4f}')

# Evaluation
model.eval()
all_predictions = []
all_labels = []
all_losses = []
with torch.no_grad():
    for inputs, labels in test_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        all_losses.append(loss.item())

        # Apply sigmoid activation to convert logits to probabilities
        probs = torch.sigmoid(outputs)

        # Convert probabilities to binary predictions
        predicted = torch.round(probs)

        all_predictions.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

# Compute metrics
accuracy = accuracy_score(all_labels, all_predictions)
precision = precision_score(all_labels, all_predictions, average='weighted')
recall = recall_score(all_labels, all_predictions, average='weighted')
f1 = f1_score(all_labels, all_predictions, average='weighted')
conf_matrix = multilabel_confusion_matrix(all_labels, all_predictions)

print(f'Accuracy: {accuracy:.4f}')
print(f'Precision: {precision:.4f}')
print(f'Recall: {recall:.4f}')
print(f'F1 Score: {f1:.4f}')
print('Confusion Matrix:')
print(conf_matrix)
print(f'Average Loss: {np.mean(all_losses):.4f}')
