from keras.optimizers import Adam
from sklearn.metrics import classification_report
import numpy as np
from model import create_model
from preprocessing import create_data_generators
from CustomEarlyStopping import CustomEarlyStoppingAndSaveBest

# Load and preprocess the dataset
dataset_path = 'C:/Users/Pana/Desktop/Northumbria/Final Year/Individual Computing Project ' \
            'KV6003BNN01/Speech-Emotion-Recognition---Audio-Dataset/models/deep learning for ' \
            'images/datasets/CREMAD/MFCCs/MFCC_32x32/'
input_shape = (32, 32, 3)
num_classes = 6
batch_size = 64
epochs = 1000
# Create data generators for training, validation, and testing
train_generator, validation_generator, test_generator = create_data_generators(
    dataset_path=dataset_path,
    target_size=input_shape[:2],
    batch_size=batch_size,
    validation_split=0.2,
    test_split=0.1,
)

# Create and compile the model
model = create_model(input_shape=input_shape, num_classes=num_classes)
model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])

# Custom early stopping is commented out; uncomment if needed
custom_early_stopping = CustomEarlyStoppingAndSaveBest(patience=20, verbose=1)

# Model training
print("Starting training...")
history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // batch_size,
    validation_data=validation_generator,
    validation_steps=validation_generator.samples // batch_size,
    epochs=epochs,
    callbacks=[custom_early_stopping]
)
print("Training completed.")

# Evaluate the model on the test set
print("Evaluating on test set...")
test_loss, test_acc = model.evaluate(test_generator, steps=test_generator.samples // batch_size)
print(f'\nTest accuracy: {test_acc*100:.2f}%')

# Predictions for classification report
y_pred = []
y_true = []
for _ in range(test_generator.samples // batch_size):
    X_test, y_test = next(test_generator)
    y_pred.extend(np.argmax(model.predict(X_test), axis=1))
    y_true.extend(np.argmax(y_test, axis=1))

# Generate classification report
print("Classification Report:")
print(classification_report(y_true, y_pred, target_names=test_generator.class_indices.keys()))
