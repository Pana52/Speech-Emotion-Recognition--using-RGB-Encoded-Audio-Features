# Import necessary libraries
import os
import numpy as np
from keras_preprocessing.image import img_to_array, load_img
from keras.applications.resnet import ResNet50, preprocess_input
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D, Input
from keras.utils import to_categorical
from keras.optimizers import Adam, SGD
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from deap import base, creator, tools, algorithms
import random
from keras.callbacks import EarlyStopping

# Constants
DATA_DIR = "C:/Users/Pana/Desktop/Northumbria/Final Year/Individual Computing Project KV6003BNN01/datasets/Mixed/EMODB/"
IMAGE_SUBFOLDER = 'images'
EMOTIONS = ['anger', 'boredom', 'disgust', 'fear', 'happiness', 'neutral', 'sadness']
NUM_CLASSES = len(EMOTIONS)
IMAGE_SIZE = (256, 256)


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


# Initialize ResNet50 model for classification with GA hyperparameters
def build_classification_model(num_classes, learning_rate=0.0001, batch_size=32, neurons=1024, activation='relu',
                               optimizer_choice='adam', input_shape=(2048,)):
    input_layer = Input(shape=input_shape)
    x = Dense(neurons, activation=activation)(input_layer)
    predictions = Dense(num_classes, activation='softmax')(x)
    model = Model(inputs=input_layer, outputs=predictions)

    if optimizer_choice == 'adam':
        optimizer = Adam(learning_rate=learning_rate)
    elif optimizer_choice == 'sgd':
        optimizer = SGD(learning_rate=learning_rate)
    # Add more optimizers as needed

    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
    return model, batch_size


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


# Custom Mutation and Crossover Functions
def mutate_individual(individual):
    individual[0] = random.choice([0.0001, 0.0005, 0.001, 0.005, 0.01])
    individual[1] = random.choice([16, 32, 64, 128])
    individual[2] = random.choice([256, 512, 1024, 2048])
    individual[3] = random.choice(['relu', 'tanh', 'sigmoid'])
    individual[4] = random.choice(['adam', 'sgd'])
    return individual,


def crossover_individual(ind1, ind2):
    cxpoint = random.randint(1, len(ind1) - 1)
    ind1[cxpoint:], ind2[cxpoint:] = ind2[cxpoint:], ind1[cxpoint:]
    return ind1, ind2


# DEAP setup for GA
def setup_ga():
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMax)

    toolbox = base.Toolbox()
    toolbox.register("attr_activation", random.choice, ['relu', 'tanh', 'sigmoid'])
    toolbox.register("attr_optimizer", random.choice, ['adam', 'sgd'])
    toolbox.register("attr_float", random.choice, [0.0001, 0.001, 0.01])
    toolbox.register("attr_int", random.choice, [16, 32, 64, 128, 256])
    toolbox.register("attr_neurons", random.choice, [128, 256, 512, 1024, 2048])

    toolbox.register("individual", tools.initCycle, creator.Individual,
                     (toolbox.attr_float, toolbox.attr_int, toolbox.attr_neurons,
                      toolbox.attr_activation, toolbox.attr_optimizer), n=1)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    toolbox.register("mate", crossover_individual)
    toolbox.register("mutate", mutate_individual)
    toolbox.register("select", tools.selTournament, tournsize=3)

    return toolbox


# Evaluation function for GA
def evaluate(individual, features, labels):
    learning_rate, batch_size, neurons = individual
    model, batch_size = build_classification_model(NUM_CLASSES, learning_rate, int(batch_size), neurons,
                                                   input_shape=(2048,))
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2)
    early_stopping = EarlyStopping(monitor='val_loss', patience=5, verbose=1)
    history = model.fit(X_train, y_train, batch_size=int(batch_size), epochs=10,
                        validation_data=(X_test, y_test), verbose=1, callbacks=[early_stopping])
    accuracy = np.max(history.history['val_accuracy'])
    return (accuracy,)


# Main function to execute the process
def main(data_dir):
    # Extract features
    feature_extractor = build_feature_extractor()
    features, labels = load_dataset_and_extract_features(data_dir, feature_extractor)

    # Prepare the GA
    toolbox = setup_ga()
    toolbox.register("evaluate", evaluate, features=features, labels=labels)

    # GA population
    population = toolbox.population(n=10)

    # Run GA
    algorithms.eaSimple(population, toolbox, cxpb=0.5, mutpb=0.2, ngen=5, verbose=True)

    best_individual = tools.selBest(population, 1)[0]
    print("Best Individual = ", best_individual)
    print("Best Hyperparameters: Learning Rate: {}, Batch Size: {}, Neurons: {}".format(best_individual[0],
                                                                                        best_individual[1],
                                                                                        best_individual[2]))

    # Retrain model with the best hyperparameters
    best_learning_rate, best_batch_size, best_neurons = best_individual
    model, _ = build_classification_model(NUM_CLASSES, best_learning_rate, best_batch_size, best_neurons,
                                          input_shape=(2048,))

    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)
    early_stopping = EarlyStopping(monitor='val_loss', patience=5, verbose=1)
    model.fit(X_train, y_train, batch_size=best_batch_size, epochs=10, validation_data=(X_test, y_test),
              callbacks=[early_stopping])

    # Evaluate the best model
    y_pred = model.predict(X_test)
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_true_classes = np.argmax(y_test, axis=1)

    # Print classification report
    print("\nClassification Report of the Best Performing Model:")
    print(classification_report(y_true_classes, y_pred_classes, target_names=EMOTIONS))


if __name__ == "__main__":
    main(DATA_DIR)
