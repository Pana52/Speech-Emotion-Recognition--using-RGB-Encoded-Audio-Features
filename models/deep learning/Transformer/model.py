from keras import layers, models
import tensorflow as tf  # Make sure to import TensorFlow


def build_transformer_model(input_shape, num_classes, num_heads=2, ff_dim=32):
    inputs = layers.Input(shape=(input_shape,))

    # Corrected usage of expand_dims
    x = layers.Lambda(lambda x: tf.expand_dims(x, axis=1))(inputs)

    # Transformer block
    attention_output = layers.MultiHeadAttention(num_heads=num_heads, key_dim=input_shape)(x, x)
    attention_output = layers.Dropout(0.1)(attention_output)
    attention_output = layers.LayerNormalization(epsilon=1e-6)(attention_output + x)

    # Feed-forward part of the Transformer
    ff_output = layers.Dense(ff_dim, activation="relu")(attention_output)
    ff_output = layers.Dense(input_shape)(ff_output)
    ff_output = layers.LayerNormalization(epsilon=1e-6)(ff_output + attention_output)

    # Flattening the output to feed into a Dense layer
    x = layers.Flatten()(ff_output)
    x = layers.Dropout(0.1)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    model = models.Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    return model
