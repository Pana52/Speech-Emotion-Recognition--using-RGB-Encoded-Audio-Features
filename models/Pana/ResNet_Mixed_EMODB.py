import numpy as np
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.callbacks import EarlyStopping

DATASET_PATH = "C:/Users/Pana/Desktop/Northumbria/Final Year/Individual Computing Project " \
               "KV6003BNN01/Speech-Emotion-Recognition---Audio-Dataset/models/Pana/"
EPOCH = 100
PATIENCE = 20
BATCH_SIZE = 32


def load_datasets(train_file, val_file, test_file):
    # Load training data
    with np.load(train_file) as data:
        X_train = data['features']
        y_train = data['labels']

    # Load validation data
    with np.load(val_file) as data:
        X_val = data['features']
        y_val = data['labels']

    # Load test data
    with np.load(test_file) as data:
        X_test = data['features']
        y_test = data['labels']

    # Encode labels to integers
    le = LabelEncoder()
    y_train_encoded = le.fit_transform(y_train)
    y_val_encoded = le.transform(y_val)
    y_test_encoded = le.transform(y_test)

    # Convert labels to one-hot encoding
    y_train_onehot = to_categorical(y_train_encoded)
    y_val_onehot = to_categorical(y_val_encoded)
    y_test_onehot = to_categorical(y_test_encoded)

    return X_train, y_train_onehot, X_val, y_val_onehot, X_test, y_test_onehot, le.classes_


def create_model(input_shape, num_classes):
    model = Sequential([
        Dense(256, activation='relu', input_shape=(input_shape,)),
        Dropout(0.5),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ])

    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model


def train_model(X_train, y_train, X_val, y_val, epochs=EPOCH, batch_size=BATCH_SIZE):
    input_shape = X_train.shape[1]  # Number of features
    num_classes = y_train.shape[1]  # Number of unique labels

    model = create_model(input_shape, num_classes)

    # Define early stopping callback
    early_stopping = EarlyStopping(monitor='val_loss', patience=PATIENCE, restore_best_weights=True)

    history = model.fit(X_train, y_train,
                        epochs=epochs,
                        batch_size=batch_size,
                        validation_data=(X_val, y_val),
                        callbacks=[early_stopping])
    return model, history


def main():
    train_file = DATASET_PATH + "train_dataset.npz"
    val_file = DATASET_PATH + "validation_dataset.npz"
    test_file = DATASET_PATH + "test_dataset.npz"

    # Load the datasets
    X_train, y_train, X_val, y_val, X_test, y_test_onehot, classes = load_datasets(train_file, val_file, test_file)

    # Train the model
    model, history = train_model(X_train, y_train, X_val, y_val)

    # Predictions on test set
    y_pred_onehot = model.predict(X_test)
    y_pred = np.argmax(y_pred_onehot, axis=1)
    y_true = np.argmax(y_test_onehot, axis=1)

    # Generate classification report
    report = classification_report(y_true, y_pred, target_names=classes)
    print(report)


if __name__ == '__main__':
    main()