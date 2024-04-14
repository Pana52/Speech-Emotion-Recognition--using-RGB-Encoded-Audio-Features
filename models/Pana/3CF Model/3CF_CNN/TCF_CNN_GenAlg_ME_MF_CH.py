# Import necessary libraries
import os
import numpy as np
from keras_preprocessing.image import img_to_array, load_img
from keras.models import Model
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
from keras.optimizers import Adam
from sklearn.metrics import classification_report
from keras.callbacks import EarlyStopping
from keras.layers import Input, Dense, GlobalAveragePooling2D, Dropout, Activation, LeakyReLU, ELU, Conv2D, MaxPooling2D
import random
# Constants
DATASET = 'EMODB'
DATA_DIR = "C:/Users/Pana/Desktop/Northumbria/Final Year/Individual Computing Project KV6003BNN01/datasets/Mixed/" + DATASET +"/256p/"
IMAGE_SUBFOLDER = 'ME_MF_CH'
MODEL = 'CNN'
EMOTIONS = ['anger', 'boredom', 'disgust', 'fear', 'happiness', 'neutral', 'sadness']
NUM_CLASSES = len(EMOTIONS)
IMAGE_SIZE = (256, 256)
BATCH_SIZE = 32
EPOCHS = 500
PATIENCE = 50


def initialize_population(pop_size):
    population = []
    for _ in range(pop_size):
        individual = {
            'learning_rate': 10 ** np.random.uniform(-10, -1),
            'batch_size': np.random.choice([16, 32, 64, 128]),
            'dense_neurons': np.random.choice([128, 256, 512, 1024]),
            'activation': np.random.choice(['relu', 'tanh', 'sigmoid', 'leaky_relu', 'elu']),
            'dropout_rate': np.random.uniform(0.01, 0.5),
            'n_clusters': np.random.randint(2, 20)
        }
        population.append(individual)
    return population


def crossover(parent1, parent2):
    child = {}
    for param in parent1:
        child[param] = parent1[param] if random.random() > 0.5 else parent2[param]
    return child


def mutate(child):
    mutation_param = random.choice(list(child.keys()))
    if mutation_param == 'learning_rate':
        child[mutation_param] = 10 ** np.random.uniform(-10, -1)
    elif mutation_param == 'batch_size':
        child[mutation_param] = np.random.choice([16, 32, 64, 128])
    elif mutation_param == 'dense_neurons':
        child[mutation_param] = np.random.choice([128, 256, 512, 1024])
    elif mutation_param == 'activation':
        child[mutation_param] = np.random.choice(['relu', 'tanh', 'sigmoid', 'leaky_relu', 'elu'])
    elif mutation_param == 'dropout_rate':
        child[mutation_param] = np.random.uniform(0.01, 0.5)
    elif mutation_param == 'n_clusters':
        child[mutation_param] = np.random.randint(2, 20)
    return child


def build_classification_model(num_classes, hyperparams):
    input_layer = Input(shape=(*IMAGE_SIZE, 3))  # Using image size directly

    # First convolutional block
    x = Conv2D(32, (3, 3), padding='same')(input_layer)
    x = Activation('relu')(x)
    x = MaxPooling2D((2, 2))(x)

    # Second convolutional block
    x = Conv2D(64, (3, 3), padding='same')(x)
    if hyperparams['activation'] == 'leaky_relu':
        x = LeakyReLU()(x)
    elif hyperparams['activation'] == 'elu':
        x = ELU()(x)
    else:
        x = Activation(hyperparams['activation'])(x)
    x = MaxPooling2D((2, 2))(x)

    # Third convolutional block
    x = Conv2D(128, (3, 3), padding='same')(x)
    x = Activation('relu')(x)  # Generally, 'relu' is safe for deep layers
    x = MaxPooling2D((2, 2))(x)

    # Global Average Pooling followed by Fully Connected Layers
    x = GlobalAveragePooling2D()(x)
    x = Dense(hyperparams['dense_neurons'])(x)
    x = Activation(hyperparams['activation'])(x)
    x = Dropout(hyperparams['dropout_rate'])(x)

    # Output layer
    predictions = Dense(num_classes, activation='softmax')(x)

    # Compile model
    model = Model(inputs=input_layer, outputs=predictions)
    model.compile(optimizer=Adam(learning_rate=hyperparams['learning_rate']),
                  loss='categorical_crossentropy', metrics=['accuracy'])
    return model


def load_dataset_and_extract_features(data_dir):

    images = []
    labels = []

    for label, emotion in enumerate(EMOTIONS):
        emotion_image_path = os.path.join(data_dir, IMAGE_SUBFOLDER, emotion)
        for image_file in os.listdir(emotion_image_path):
            full_image_path = os.path.join(emotion_image_path, image_file)
            img = load_img(full_image_path, target_size=IMAGE_SIZE)
            img = img_to_array(img)
            img = img / 255.0  # Normalization step, adjust according to your model's requirements
            images.append(img)
            labels.append(label)

    images = np.array(images)
    labels = to_categorical(labels, num_classes=NUM_CLASSES)
    return images, labels


def train_and_evaluate(hyperparams, images, labels):
    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=42)

    # Build the 3CF_CNN model using hyperparameters from the genetic algorithm
    model = build_classification_model(NUM_CLASSES, hyperparams)

    # Early stopping to prevent overfitting
    early_stopping = EarlyStopping(monitor='val_loss', patience=PATIENCE, verbose=0, restore_best_weights=True)

    # Train the model
    model.fit(X_train, y_train, batch_size=hyperparams['batch_size'], epochs=EPOCHS,
              validation_data=(X_test, y_test), callbacks=[early_stopping], verbose=0)

    # Evaluate the model on the testing set
    _, accuracy = model.evaluate(X_test, y_test, verbose=0)

    # Get predictions for further evaluation if necessary
    predictions = model.predict(X_test)
    y_pred_labels = np.argmax(predictions, axis=1)
    y_true_labels = np.argmax(y_test, axis=1)

    return accuracy, y_true_labels, y_pred_labels


def main():
    POPULATION_SIZE = 30
    NUM_GENERATIONS = 20
    NUM_PARENTS = 5

    # Initialize population with hyperparameters
    population = initialize_population(POPULATION_SIZE)

    # Load dataset (assuming raw images and labels are prepared correctly)
    images, labels = load_dataset_and_extract_features(DATA_DIR)  # Adjusted function needed

    best_accuracy = 0
    best_hyperparams = None
    best_y_true = None
    best_y_pred = None

    for generation in range(NUM_GENERATIONS):
        print(f"Generation {generation + 1}")

        generation_accuracies = []
        for individual in population:
            # Updated to directly use the images and labels
            accuracy, y_true_labels, y_pred_labels = train_and_evaluate(individual, images, labels)
            generation_accuracies.append(accuracy)

            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_hyperparams = individual
                best_y_true = y_true_labels
                best_y_pred = y_pred_labels

        # Selection based on accuracy
        sorted_indices = np.argsort(generation_accuracies)[::-1]
        top_individuals = [population[i] for i in sorted_indices[:NUM_PARENTS]]

        # Crossover and mutation to create the next generation
        next_generation = []
        for _ in range(POPULATION_SIZE - len(top_individuals)):
            parent1, parent2 = random.sample(top_individuals, 2)
            child = crossover(parent1, parent2)
            child = mutate(child)
            next_generation.append(child)

        population = top_individuals + next_generation

    # Redirecting output to a text file
    with open("CNN_" + IMAGE_SUBFOLDER + "_" + DATASET + "_" + MODEL + "_optimization_results.txt", "w") as output_file:
        print(f"Optimization completed. Best Accuracy: {best_accuracy}", file=output_file)
        print(f"Best Hyperparameters: {best_hyperparams}", file=output_file)

        # Print classification report for the best model
        if best_y_true is not None and best_y_pred is not None:
            print("Classification Report for the Best Model:", file=output_file)
            print(classification_report(best_y_true, best_y_pred, target_names=EMOTIONS), file=output_file)


if __name__ == "__main__":
    main()
