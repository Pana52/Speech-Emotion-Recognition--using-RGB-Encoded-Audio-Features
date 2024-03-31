from keras.models import Model
from keras.layers import Input, Conv1D, BatchNormalization, Activation, Add, MaxPooling1D, Flatten, Dense


def residual_block(input_tensor, filters, kernel_size=3, stride=1):
    """A ResNet-style residual block for 1D data."""
    # Main path
    x = Conv1D(filters, kernel_size=kernel_size, strides=stride, padding='same')(input_tensor)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = Conv1D(filters, kernel_size=kernel_size, strides=1, padding='same')(x)
    x = BatchNormalization()(x)

    # Shortcut path
    shortcut = input_tensor
    if stride != 1 or input_tensor.shape[-1] != filters:
        shortcut = Conv1D(filters, kernel_size=1, strides=stride, padding='same')(input_tensor)
        shortcut = BatchNormalization()(shortcut)

    # Merge paths
    x = Add()([x, shortcut])
    x = Activation('relu')(x)
    return x


def build_resnet(input_shape, num_classes):
    """Builds a ResNet-like model for 1D data."""
    inputs = Input(shape=input_shape)

    x = Conv1D(64, kernel_size=7, strides=2, padding='same')(inputs)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling1D(pool_size=3, strides=2, padding='same')(x)

    # Example ResNet blocks
    x = residual_block(x, filters=32, stride=1)
    x = residual_block(x, filters=32, stride=1)

    x = residual_block(x, filters=64, stride=2)
    x = residual_block(x, filters=64, stride=1)

    x = residual_block(x, filters=128, stride=2)
    x = residual_block(x, filters=128, stride=1)

    x = Flatten()(x)
    x = Dense(512, activation='relu')(x)
    outputs = Dense(num_classes, activation='softmax')(x)

    model = Model(inputs=inputs, outputs=outputs)

    return model
