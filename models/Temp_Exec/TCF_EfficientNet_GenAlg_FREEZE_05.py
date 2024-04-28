import os
import numpy as np
from keras import Input
from keras_preprocessing.image import img_to_array, load_img
from keras.applications.efficientnet import EfficientNetB0, preprocess_input
from keras.models import Model
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
from keras.optimizers import Adam
from sklearn.metrics import classification_report
from sklearn.preprocessing import OneHotEncoder
from keras.callbacks import EarlyStopping
from sklearn.cluster import KMeans
from keras.layers import Dense, GlobalAveragePooling2D, Dropout, Activation, LeakyReLU, ELU
import random

# Constants
DATASET = 'RAVDESS'
DATA_DIR = f"C:/Users/Pana/Desktop/Northumbria/Final Year/Individual Computing Project KV6003BNN01/datasets/Mixed/{DATASET}/256p/3CF/"
IMAGE_SUBFOLDER = 'ME_MF_CH'
MODEL = 'EFFICIENTNET'
MODE = 'FREEZE'
# ALL CLASSES
# EMODB
# EMOTIONS = ['anger', 'boredom', 'disgust', 'fear', 'happiness', 'neutral', 'sadness']
# RAVDESS
EMOTIONS = ['angry', 'calm', 'disgust', 'fearful', 'happy', 'neutral', 'sad', 'surprised']

# 4 CLASSES
# EMODB
# EMOTIONS = ['anger', 'happiness', 'neutral', 'sadness']
# RAVDESS
# EMOTIONS = ['angry', 'happy', 'neutral', 'sad']
NUM_CLASSES = len(EMOTIONS)
IMAGE_SIZE = (256, 256)
BATCH_SIZE = 32
EPOCHS = 100
PATIENCE = 20


def initialize_population(pop_size):
    population = []
    for _ in range(pop_size):
        individual = {
            'learning_rate': 10 ** np.random.uniform(-5, -1),
            'batch_size': np.random.choice([16, 32, 64, 128]),
            'dense_neurons': np.random.choice([64, 128, 256, 512]),
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
        child[mutation_param] = 10 ** np.random.uniform(-5, -1)
    elif mutation_param == 'batch_size':
        child[mutation_param] = np.random.choice([16, 32, 64, 128])
    elif mutation_param == 'dense_neurons':
        child[mutation_param] = np.random.choice([64, 128, 256, 512])
    elif mutation_param == 'activation':
        child[mutation_param] = np.random.choice(['relu', 'tanh', 'sigmoid', 'leaky_relu', 'elu'])
    elif mutation_param == 'dropout_rate':
        child[mutation_param] = np.random.uniform(0.01, 0.5)
    elif mutation_param == 'n_clusters':
        child[mutation_param] = np.random.randint(2, 20)
    return child


def load_and_extract_features(img_path, feature_model):
    img = load_img(img_path, target_size=IMAGE_SIZE)
    img = img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = preprocess_input(img)
    features = feature_model.predict(img)
    return np.squeeze(features).flatten()  # Ensure features are flattened


def build_feature_extractor():
    base_model = EfficientNetB0(weights='imagenet', include_top=False, input_shape=(*IMAGE_SIZE, 3))
    model = Model(inputs=base_model.input, outputs=GlobalAveragePooling2D()(base_model.output))
    return model


def augment_features_with_clusters(features, cluster_labels):
    # One-hot encode cluster labels
    encoder = OneHotEncoder(sparse=False)
    cluster_labels_one_hot = encoder.fit_transform(cluster_labels.reshape(-1, 1))

    # Concatenate original features with one-hot encoded cluster labels
    # Ensure the final shape is suitable for the model input
    augmented_features = np.concatenate([features.reshape(features.shape[0], -1), cluster_labels_one_hot], axis=1)
    return augmented_features


def apply_clustering(features, n_clusters):
    if features.ndim != 2:
        features = features.reshape(features.shape[0], -1)  # Reshape to 2D if necessary
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    cluster_labels = kmeans.fit_predict(features)
    return cluster_labels, kmeans.cluster_centers_


def build_classification_model(num_classes, feature_length, hyperparams):
    input_layer = Input(shape=(feature_length,))
    x = Dense(hyperparams['dense_neurons'], activation=None)(input_layer)
    if hyperparams['activation'] == 'leaky_relu':
        x = LeakyReLU()(x)
    elif hyperparams['activation'] == 'elu':
        x = ELU()(x)
    else:
        x = Activation(hyperparams['activation'])(x)

    x = Dropout(hyperparams['dropout_rate'])(x)
    predictions = Dense(num_classes, activation='softmax')(x)
    model = Model(inputs=input_layer, outputs=predictions)
    model.compile(optimizer=Adam(learning_rate=hyperparams['learning_rate']),
                  loss='categorical_crossentropy', metrics=['accuracy'])
    return model


def load_dataset_and_extract_features(data_dir, feature_model):
    features = []
    labels = []

    for emotion in EMOTIONS:
        emotion_image_path = os.path.join(data_dir, IMAGE_SUBFOLDER, emotion)
        for image_file in os.listdir(emotion_image_path):
            full_image_path = os.path.join(emotion_image_path, image_file)
            feature = load_and_extract_features(full_image_path, feature_model)
            features.append(feature)  # Ensure that each feature is flat
            labels.append(EMOTIONS.index(emotion))

    features = np.array(features)
    if features.ndim > 2:
        features = features.reshape(features.shape[0], -1)  # Additional check to ensure 2D
    labels = to_categorical(labels, num_classes=NUM_CLASSES)
    return features, labels


def train_and_evaluate(hyperparams, features, labels, unfreeze=False, layers_to_unfreeze=3):
    # Apply clustering
    cluster_labels, _ = apply_clustering(features, hyperparams['n_clusters'])
    augmented_features = augment_features_with_clusters(features, cluster_labels)

    # Split data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(augmented_features, labels, test_size=0.2, random_state=42)

    # Prepare the correct input shape for the model, it must match the augmented feature shape
    model_input_shape = X_train.shape[1]  # This should capture the correct feature length

    # Update model creation to handle new input shape and include unfreezing logic
    model = build_classification_model(NUM_CLASSES, model_input_shape, hyperparams)

    # Set up early stopping to prevent overfitting
    early_stopping = EarlyStopping(monitor='val_loss', patience=PATIENCE, verbose=0, restore_best_weights=True)
    model.fit(X_train, y_train, batch_size=hyperparams['batch_size'], epochs=EPOCHS, validation_data=(X_test, y_test),
              callbacks=[early_stopping], verbose=0)

    # Evaluate the model on the test set
    _, accuracy = model.evaluate(X_test, y_test, verbose=0)

    # Get predictions and labels for accuracy comparison
    predictions = model.predict(X_test)
    y_pred_labels = np.argmax(predictions, axis=1)
    y_true_labels = np.argmax(y_test, axis=1)

    return accuracy, y_true_labels, y_pred_labels


def main():
    POPULATION_SIZE = 10
    NUM_GENERATIONS = 20
    NUM_PARENTS = 5

    feature_model = EfficientNetB0(weights='imagenet', include_top=False, input_shape=(*IMAGE_SIZE, 3))
    features, labels = load_dataset_and_extract_features(DATA_DIR, feature_model)

    population = initialize_population(POPULATION_SIZE)
    best_accuracy = 0
    best_hyperparams = None
    best_y_true = None
    best_y_pred = None

    for generation in range(NUM_GENERATIONS):
        print(f"Generation {generation + 1}")

        generation_accuracies = []
        for individual in population:
            accuracy, y_true_labels, y_pred_labels = train_and_evaluate(individual, features, labels)
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

    with open(f"{IMAGE_SUBFOLDER}_{DATASET}_{MODEL}_{MODE}_optimization_results.txt", "w") as output_file:
        print(f"Optimization completed. Best Accuracy: {best_accuracy}", file=output_file)
        print(f"Best Hyperparameters: {best_hyperparams}", file=output_file)

        # Print classification report for the best model
        if best_y_true is not None and best_y_pred is not None:
            print("Classification Report for the Best Model:", file=output_file)
            print(classification_report(best_y_true, best_y_pred, target_names=EMOTIONS), file=output_file)


if __name__ == "__main__":
    main()
