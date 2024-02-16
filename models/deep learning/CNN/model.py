from keras.models import Sequential
from keras.layers import Dense, Conv1D, MaxPooling1D, Flatten, Dropout, BatchNormalization, Activation
from keras.optimizers import Adam
from keras.regularizers import l1_l2


def create_model(input_shape, num_classes):
    model = Sequential([
        Conv1D(64, kernel_size=3, padding='same', input_shape=input_shape, kernel_regularizer=l1_l2(l1=1e-5, l2=1e-4)),
        BatchNormalization(),
        Activation('relu'),
        MaxPooling1D(pool_size=2),
        Conv1D(128, kernel_size=3, padding='same', kernel_regularizer=l1_l2(l1=1e-5, l2=1e-4)),
        BatchNormalization(),
        Activation('relu'),
        MaxPooling1D(pool_size=2),
        Flatten(),
        Dense(128, activation='relu', kernel_regularizer=l1_l2(l1=1e-5, l2=1e-4)),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ])

    model.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])
    return model
