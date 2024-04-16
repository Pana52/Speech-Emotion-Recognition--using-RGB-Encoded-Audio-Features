import os
import numpy as np
import tensorflow as tf
from keras import Input
from keras.applications.convnext import preprocess_input
from keras_preprocessing.image import img_to_array, load_img
from keras.models import Model
from keras.layers import MaxPooling2D, GlobalAveragePooling2D, Dense, Dropout, BatchNormalization, multiply, Add, Conv2D, UpSampling2D
from keras.regularizers import l2
from keras.utils import to_categorical
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.utils.class_weight import compute_class_weight

# Constants
DATASET = 'EMODB'
IMAGE_SUBFOLDER = 'CH_ME_MF'
DATA_DIR = f"C:/Users/Pana/Desktop/Northumbria/Final Year/Individual Computing Project KV6003BNN01/datasets/Mixed/{DATASET}/Augmented/256p/{IMAGE_SUBFOLDER}"
MODEL = 'CNN'
EMOTIONS = ['anger', 'boredom', 'disgust', 'fear', 'happiness', 'neutral', 'sadness']
NUM_CLASSES = len(EMOTIONS)
IMAGE_SIZE = (256, 256)
BATCH_SIZE = 32
EPOCHS = 500
PATIENCE = 50
LEARNING_RATE = 0.001


class ChannelAttention(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(ChannelAttention, self).__init__(**kwargs)

    def build(self, input_shape):
        self.dense1 = Dense(input_shape[-1] // 8, activation='relu', kernel_initializer='he_normal', use_bias=False)
        self.dense2 = Dense(input_shape[-1], activation='sigmoid', kernel_initializer='he_normal', use_bias=False)
        super(ChannelAttention, self).build(input_shape)

    def call(self, inputs):
        x = GlobalAveragePooling2D()(inputs)
        x = self.dense1(x)
        x = self.dense2(x)
        return multiply([inputs, x])

    def compute_output_shape(self, input_shape):
        return input_shape


def build_cnn():
    model_input = Input(shape=(*IMAGE_SIZE, 3))
    x = Conv2D(32, (3, 3), activation='relu', padding='same', kernel_regularizer=l2(0.01))(model_input)
    x = BatchNormalization()(x)
    x = MaxPooling2D((2, 2))(x)
    f1 = Conv2D(64, (3, 3), activation='relu', padding='same', kernel_regularizer=l2(0.01))(x)
    f1 = BatchNormalization()(f1)
    f1_pool = MaxPooling2D((2, 2))(f1)
    f2 = Conv2D(128, (3, 3), activation='relu', padding='same', kernel_regularizer=l2(0.01))(f1_pool)
    f2 = BatchNormalization()(f2)
    f2_pool = MaxPooling2D((2, 2))(f2)
    f2_adjusted = Conv2D(64, (1, 1), activation='relu', padding='same')(f2_pool)
    f2_upsampled = UpSampling2D(size=(2, 2))(f2_adjusted)
    f1_combined = Add()([f2_upsampled, f1_pool])
    f1_attention = ChannelAttention()(f1_combined)
    x = GlobalAveragePooling2D()(f1_attention)
    feature_model = Model(inputs=model_input, outputs=x)
    return feature_model


def build_classification_model(num_classes, input_shape=(64,)):
    input_layer = Input(shape=input_shape)
    x = Dense(128, activation='relu')(input_layer)
    x = Dropout(0.5)(x)
    predictions = Dense(num_classes, activation='softmax')(x)
    model = Model(inputs=input_layer, outputs=predictions)
    model.compile(optimizer=Adam(learning_rate=LEARNING_RATE), loss='categorical_crossentropy', metrics=['accuracy'])
    return model


def load_dataset_and_extract_features(data_dir, feature_model):
    features = []
    labels = []
    for emotion in EMOTIONS:
        emotion_image_path = os.path.join(data_dir, emotion)
        for image_file in os.listdir(emotion_image_path):
            full_image_path = os.path.join(emotion_image_path, image_file)
            img = load_img(full_image_path, target_size=IMAGE_SIZE)
            img = img_to_array(img)
            img = np.expand_dims(img, axis=0)
            img = preprocess_input(img)
            img_features = feature_model.predict(img)
            features.append(np.squeeze(img_features))
            labels.append(EMOTIONS.index(emotion))
    features = np.array(features)
    labels = np.array(labels)
    labels_one_hot = to_categorical(labels, num_classes=NUM_CLASSES)
    return features, labels_one_hot, labels


def main(data_dir):
    feature_model = build_cnn()
    features, labels_one_hot, original_labels = load_dataset_and_extract_features(data_dir, feature_model)
    classification_model = build_classification_model(NUM_CLASSES, input_shape=(64,))
    class_weights = compute_class_weight('balanced', classes=np.unique(original_labels), y=original_labels)
    class_weight_dict = dict(enumerate(class_weights))
    X_train, X_test, y_train, y_test = train_test_split(features, labels_one_hot, test_size=0.2, random_state=42)
    early_stopping = EarlyStopping(monitor='val_loss', patience=PATIENCE, verbose=1, restore_best_weights=True)
    model_checkpoint = ModelCheckpoint('best_model.h5', monitor='val_accuracy', mode='max', verbose=1,
                                       save_best_only=True)
    classification_model.fit(X_train, y_train, batch_size=BATCH_SIZE, epochs=EPOCHS, validation_data=(X_test, y_test),
                             callbacks=[early_stopping, model_checkpoint], class_weight=class_weight_dict)
    y_pred = classification_model.predict(X_test)
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_true_classes = np.argmax(y_test, axis=1)
    report = classification_report(y_true_classes, y_pred_classes, target_names=EMOTIONS)
    output_filename = f"CNN_{IMAGE_SUBFOLDER}_{DATASET}_{MODEL}_optimization_results.txt"
    with open(output_filename, "w") as output_file:
        output_file.write(report)
    print(f"Classification report saved to {output_filename}")


if __name__ == "__main__":
    main(DATA_DIR)
