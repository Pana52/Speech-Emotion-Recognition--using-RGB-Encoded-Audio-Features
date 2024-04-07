# Import necessary libraries
import os
import numpy as np
from keras_preprocessing.image import img_to_array, load_img
from keras.applications.resnet import ResNet50, preprocess_input
from keras.models import Model
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
from keras.optimizers import Adam
from sklearn.metrics import classification_report
from keras.callbacks import EarlyStopping
from sklearn.cluster import KMeans
from keras.layers import Input, Dense, GlobalAveragePooling2D
import random

# Constants
DATA_DIR = "C:/Users/Pana/Desktop/Northumbria/Final Year/Individual Computing Project KV6003BNN01/datasets/Mixed/EMODB/"
IMAGE_SUBFOLDER = '3CF_Images'
EMOTIONS = ['anger', 'boredom', 'disgust', 'fear', 'happiness', 'neutral', 'sadness']
NUM_CLASSES = len(EMOTIONS)
IMAGE_SIZE = (256, 256)  # Adjust based on your images' dimensions
BATCH_SIZE = 32
EPOCHS = 500
PATIENCE = 50
LEARNING_RATE = 0.0001
N_CLUSTERS = 10  # Number of clusters for K-means


# Population Initialization: Generate a population of hyperparameter sets.
def initialize_population(pop_size):
    population = []
    for _ in range(pop_size):
        individual = {
            'learning_rate': 10 ** np.random.uniform(-5, -3),
            'batch_size': np.random.choice([16, 32, 64]),
            'dense_neurons': np.random.choice([512, 1024, 2048])
        }
        population.append(individual)
    return population


# Selection: Select individuals based on fitness (higher is better).
def select_parents(population, fitness_scores, num_parents):
    parents = list(
        np.random.choice(population, size=num_parents, replace=False, p=fitness_scores / np.sum(fitness_scores)))
    return parents


# Crossover: Combine parents to create a new individual.
def crossover(parent1, parent2):
    child = {}
    for param in parent1:
        child[param] = parent1[param] if random.random() < 0.5 else parent2[param]
    return child


# Mutation: Randomly change one hyperparameter of an individual.
def mutate(child):
    mutation_param = random.choice(list(child.keys()))
    if mutation_param == 'learning_rate':
        child[mutation_param] = 10 ** np.random.uniform(-5, -3)
    elif mutation_param == 'batch_size':
        child[mutation_param] = np.random.choice([16, 32, 64])
    elif mutation_param == 'dense_neurons':
        child[mutation_param] = np.random.choice([512, 1024, 2048])
    return child


# Function to load, preprocess images, and extract features
def load_and_extract_features(img_path, feature_model):
    img = load_img(img_path, target_size=IMAGE_SIZE)
    img = img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = preprocess_input(img)
    features = feature_model.predict(img)
    return np.squeeze(features)


# Function to build a feature extraction model based on ResNet50
def build_feature_extractor():
    base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(*IMAGE_SIZE, 3))
    model = Model(inputs=base_model.input, outputs=GlobalAveragePooling2D()(base_model.output))
    return model


# Function to apply clustering on the extracted features
def apply_clustering(features):
    kmeans = KMeans(n_clusters=N_CLUSTERS, random_state=42)
    cluster_labels = kmeans.fit_predict(features)
    return cluster_labels, kmeans.cluster_centers_


# Initialize ResNet50 model for classification
def build_classification_model(num_classes, input_shape, learning_rate, dense_neurons):
    input_layer = Input(shape=input_shape)
    x = Dense(dense_neurons, activation='relu')(input_layer)
    predictions = Dense(num_classes, activation='softmax')(x)
    model = Model(inputs=input_layer, outputs=predictions)
    model.compile(optimizer=Adam(learning_rate=learning_rate), loss='categorical_crossentropy', metrics=['accuracy'])
    return model


# Load dataset and extract features
def load_dataset_and_extract_features(data_dir, feature_model):
    features = []
    labels = []

    for emotion in EMOTIONS:
        print(f"Processing {emotion} images...")
        emotion_image_path = os.path.join(data_dir, IMAGE_SUBFOLDER, emotion)
        for image_file in os.listdir(emotion_image_path):
            full_image_path = os.path.join(emotion_image_path, image_file)
            img_features = load_and_extract_features(full_image_path, feature_model)
            features.append(img_features)
            labels.append(EMOTIONS.index(emotion))

    features = np.array(features)
    labels = to_categorical(labels, num_classes=NUM_CLASSES)

    return features, labels


def train_and_evaluate(feature_model, hyperparams, features, labels):
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)
    # Build the model with current hyperparameters
    model = build_classification_model(NUM_CLASSES, (2048,), hyperparams['learning_rate'], hyperparams['dense_neurons'])
    # Early Stopping
    early_stopping = EarlyStopping(monitor='val_loss', patience=PATIENCE, verbose=0, restore_best_weights=True)
    # Train the model
    history = model.fit(X_train, y_train, batch_size=hyperparams['batch_size'], epochs=EPOCHS,
                        validation_data=(X_test, y_test), callbacks=[early_stopping], verbose=0)
    # Evaluate the model
    _, accuracy = model.evaluate(X_test, y_test, verbose=0)
    return accuracy


# Main function to execute the process
def main():
    # Constants for the GA
    POPULATION_SIZE = 10
    NUM_GENERATIONS = 5
    NUM_PARENTS = 5

    # Initialize the population
    population = initialize_population(POPULATION_SIZE)

    # Load dataset and extract features once to avoid repetition
    feature_model = build_feature_extractor()
    features, labels = load_dataset_and_extract_features(DATA_DIR, feature_model)

    for generation in range(NUM_GENERATIONS):
        print(f"Generation {generation + 1}/{NUM_GENERATIONS}")

        # Evaluate the population
        fitness_scores = np.array(
            [train_and_evaluate(feature_model, individual, features, labels) for individual in population])

        # Selection
        parents_indices = np.argsort(-fitness_scores)[:NUM_PARENTS]
        parents = [population[index] for index in parents_indices]

        # Crossover and Mutation to create next generation
        next_generation = []
        for _ in range(POPULATION_SIZE):
            parent1, parent2 = random.sample(parents, 2)
            child = crossover(parent1, parent2)
            child = mutate(child)
            next_generation.append(child)

        population = next_generation

    print("GA optimization completed.")


if __name__ == "__main__":
    main()
