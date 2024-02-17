import torch
import torch.nn as nn
import torch.optim as optim
from preprocessing import get_dataloaders
from model import initialize_model
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, confusion_matrix
import numpy as np

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def train_model(model, dataloaders, criterion, optimizer, num_epochs=25):
    for epoch in range(num_epochs):
        print(f'Epoch {epoch + 1}/{num_epochs}')
        print('-' * 10)

        # Variables to gather full outputs and labels for calculating metrics later
        all_labels = []
        all_preds = []

        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0

            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

                # Save labels and predictions for calculating metrics on the entire dataset
                if phase == 'val':
                    all_labels.extend(labels.cpu().numpy())
                    all_preds.extend(preds.cpu().numpy())

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)

            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            if phase == 'val':
                # Calculate additional metrics for the validation phase
                all_labels = np.array(all_labels)
                all_preds = np.array(all_preds)
                print(f'Precision: {precision_score(all_labels, all_preds, average="weighted"):.4f}')
                print(f'Recall: {recall_score(all_labels, all_preds, average="weighted"):.4f}')
                print(f'F1-Score: {f1_score(all_labels, all_preds, average="weighted"):.4f}')
                # Generating and displaying the confusion matrix

    return model


def evaluate_on_test(model, test_loader, criterion):
    model.eval()  # Set model to evaluation mode

    running_loss = 0.0
    all_preds = []
    all_labels = []

    # Disabling gradient calculation
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            _, preds = torch.max(outputs, 1)
            running_loss += loss.item() * inputs.size(0)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    test_loss = running_loss / len(test_loader.dataset)
    test_accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average="weighted")
    recall = recall_score(all_labels, all_preds, average="weighted")
    f1 = f1_score(all_labels, all_preds, average="weighted")
    confusion = confusion_matrix(all_labels, all_preds)

    print("\nTest Loss: {:.4f}".format(test_loss))
    print("Test Accuracy: {:.4f}".format(test_accuracy))
    print("Precision: {:.4f}".format(precision))
    print("Recall: {:.4f}".format(recall))
    print("F1-Score: {:.4f}".format(f1))
    print("Confusion Matrix:\n", confusion)


def main():
    train_loader, val_loader, test_loader = get_dataloaders(batch_size=16, max_len=216, augment=True)
    dataloaders = {'train': train_loader, 'val': val_loader}

    model = initialize_model().to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    trained_model = train_model(model, dataloaders, criterion, optimizer, num_epochs=25)

    # Evaluate on the test set after training is complete
    evaluate_on_test(trained_model, test_loader, criterion)


if __name__ == '__main__':
    main()
