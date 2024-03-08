from model import create_image_model
from preprocessing import create_generators
from keras.callbacks import ModelCheckpoint
from sklearn.metrics import classification_report
import numpy as np

# Define model parameters
input_shape = (224, 224, 1)  # Example input shape, adjust based on your actual data
num_classes = 6  # Update based on the actual number of classes in your dataset
batch_size = 32
epochs = 50


def train_and_evaluate_model():
    # Create data generators
    train_generator, validation_generator, test_generator = create_generators(batch_size=batch_size)

    # Initialize the model
    model = create_image_model(input_shape, num_classes)

    # Model checkpoint to save the best model
    checkpoint_cb = ModelCheckpoint("best_model.h5", save_best_only=True)

    # Training the model
    model.fit(
        train_generator,
        steps_per_epoch=len(train_generator),
        epochs=epochs,
        validation_data=validation_generator,
        validation_steps=len(validation_generator),
        callbacks=[checkpoint_cb])

    # Evaluate the model on the test set
    model.evaluate(test_generator, steps=len(test_generator))

    # Generate predictions for all samples in the test set and accumulate true labels
    test_generator.reset()  # Ensure we're starting iteration from the beginning
    predictions = model.predict(test_generator, steps=len(test_generator))
    predicted_classes = np.argmax(predictions, axis=1)

    # Get true labels from the generator
    true_labels = test_generator.classes

    # Ensure predicted_classes and true_labels are aligned
    if len(predicted_classes) != len(true_labels):
        true_labels = true_labels[:len(predicted_classes)]

    # Generate and print classification report
    print(classification_report(true_labels, predicted_classes, target_names=list(test_generator.class_indices.keys())))


if __name__ == "__main__":
    train_and_evaluate_model()
