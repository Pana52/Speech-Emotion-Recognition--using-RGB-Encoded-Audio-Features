from keras import models, layers


def create_model(input_shape, num_classes):
    """Builds a simple MLP model for audio classification.

    Parameters:
    - input_shape: Tuple, shape of the input features.
    - num_classes: Int, number of classes for classification.

    Returns:
    - model: A compiled Keras model.
    """
    model = models.Sequential([
        layers.Dense(128, activation='relu', input_shape=input_shape),
        layers.Dropout(0.5),
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(32, activation='relu'),
        layers.Dense(num_classes, activation='softmax')
    ])

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    return model
