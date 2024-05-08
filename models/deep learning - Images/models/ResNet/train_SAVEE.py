import os
import numpy as np
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import ImageDataGenerator
from keras.applications.resnet import ResNet50, preprocess_input
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D, Dropout
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping
from sklearn.metrics import classification_report
import shutil

TRAIN = 0.7
VAL = 0.2
TEST = 1 - TRAIN - VAL

IMG_SHAPE = (256, 256)
INPUT_SHAPE = (256, 256, 3)
BATCH_sIZE = 32
EPOCHS = 1000
PATIENCE = 100
NUM_CLASSES = 7


def split_dataset(dataset_path, split_train=TRAIN, split_val=VAL, split_test=TEST):
    assert split_train + split_val + split_test == 1, "Splits must sum to 1"
    # Paths for the split directories
    train_dir = os.path.join(dataset_path, 'train')
    val_dir = os.path.join(dataset_path, 'val')
    test_dir = os.path.join(dataset_path, 'test')

    # Create split directories if they do not exist
    for path in [train_dir, val_dir, test_dir]:
        if not os.path.exists(path):
            os.makedirs(path)

    # Move files into the split directories
    for class_dir in os.listdir(dataset_path):
        class_path = os.path.join(dataset_path, class_dir)
        if os.path.isdir(class_path) and class_dir not in ['train', 'val', 'test']:
            files = [os.path.join(class_path, f) for f in os.listdir(class_path)]
            train_files, test_files = train_test_split(files, test_size=split_test + split_val, random_state=42)
            val_files, test_files = train_test_split(test_files, test_size=split_test / (split_test + split_val),
                                                     random_state=42)

            # Function to copy files to their respective directories
            def copy_files(files, destination):
                for f in files:
                    dest_path = os.path.join(destination, class_dir)
                    if not os.path.exists(dest_path):
                        os.makedirs(dest_path)
                    shutil.copy(f, dest_path)

            # Copy files to respective directories
            copy_files(train_files, train_dir)
            copy_files(val_files, val_dir)
            copy_files(test_files, test_dir)


def load_data(dataset_path, img_shape, batch_size):
    train_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)
    validation_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)
    test_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

    train_generator = train_datagen.flow_from_directory(
        os.path.join(dataset_path, 'train'),
        target_size=img_shape,
        batch_size=batch_size,
        class_mode='categorical')

    validation_generator = validation_datagen.flow_from_directory(
        os.path.join(dataset_path, 'val'),
        target_size=img_shape,
        batch_size=batch_size,
        class_mode='categorical')

    test_generator = test_datagen.flow_from_directory(
        os.path.join(dataset_path, 'test'),
        target_size=img_shape,
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=False)

    return train_generator, validation_generator, test_generator


def build_model(input_shape, num_classes):
    base_model = ResNet50(include_top=False, weights='imagenet', input_shape=input_shape)
    base_model.trainable = False

    x = GlobalAveragePooling2D()(base_model.output)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.5)(x)
    predictions = Dense(num_classes, activation='softmax')(x)
    model = Model(inputs=base_model.input, outputs=predictions)

    model.compile(optimizer=Adam(lr=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])
    return model


def train_model(model, train_generator, validation_generator, epochs, patience):
    early_stopping = EarlyStopping(monitor='val_loss', patience=patience, verbose=1, restore_best_weights=True)
    model.fit(train_generator, epochs=epochs, validation_data=validation_generator, callbacks=[early_stopping])


def evaluate_model(model, test_generator):
    predictions = model.predict(test_generator, verbose=1)
    predicted_classes = np.argmax(predictions, axis=1)
    true_classes = test_generator.classes
    class_labels = list(test_generator.class_indices.keys())
    report = classification_report(true_classes, predicted_classes, target_names=class_labels)
    print(report)


def main():
    dataset_path = 'PATH'

    split_dataset(dataset_path)

    train_generator, validation_generator, test_generator = load_data(dataset_path, IMG_SHAPE, BATCH_sIZE)
    model = build_model(INPUT_SHAPE, NUM_CLASSES)
    train_model(model, train_generator, validation_generator, EPOCHS, PATIENCE)
    evaluate_model(model, test_generator)


if __name__ == "__main__":
    main()
