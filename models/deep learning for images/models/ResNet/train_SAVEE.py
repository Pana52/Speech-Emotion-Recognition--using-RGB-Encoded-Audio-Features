import os
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from keras.applications.resnet import ResNet50, preprocess_input
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D, Dropout
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping
from sklearn.metrics import classification_report


def load_data(dataset_path, img_width, img_height, batch_size):
    train_datagen = ImageDataGenerator(preprocessing_function=preprocess_input, validation_split=0.2)
    train_generator = train_datagen.flow_from_directory(
        dataset_path,
        target_size=(img_height, img_width),
        batch_size=batch_size,
        class_mode='categorical',
        subset='training')

    validation_generator = train_datagen.flow_from_directory(
        dataset_path,
        target_size=(img_height, img_width),
        batch_size=batch_size,
        class_mode='categorical',
        subset='validation')

    test_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)
    test_generator = test_datagen.flow_from_directory(
        dataset_path,
        target_size=(img_height, img_width),
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=False)

    return train_generator, validation_generator, test_generator


def build_model(img_height, img_width, num_classes):
    base_model = ResNet50(include_top=False, weights='imagenet', input_shape=(img_height, img_width, 3))
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
    dataset_path = 'C:/Users/Pana/Desktop/Northumbria/Final Year/Individual Computing Project ' \
                   'KV6003BNN01/Speech-Emotion-Recognition---Audio-Dataset/models/deep learning for ' \
                   'images/datasets/SAVEE/MFCCs/MFCC_32x32/'
    img_width, img_height = 32, 32
    batch_size = 32
    epochs = 1000
    patience = 30
    num_classes = len(next(os.walk(dataset_path))[1])

    train_generator, validation_generator, test_generator = load_data(dataset_path, img_width, img_height, batch_size)
    model = build_model(img_height, img_width, num_classes)
    train_model(model, train_generator, validation_generator, epochs, patience)
    evaluate_model(model, test_generator)


if __name__ == "__main__":
    main()
