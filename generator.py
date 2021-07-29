import tensorflow_addons as tfa
import tensorflow as tf

def resnet(filters, inputLayer):
    initalize = tf.keras.initializers.RandomNormal(stddev=0.02)

    x = tf.keras.layers.Conv2D(filters, (3,3), padding='same', kernel_initializer=initalize)(inputLayer)
    x = tfa.layers.InstanceNormalization(axis=-1)(x)
    x = tf.keras.activations.relu()(x)

    x = tf.keras.layers.Conv2D(filters, (3,3), padding='same', kernel_initializer=initalize)(x)
    x = tfa.layers.InstanceNormalization(axis=-1)(x)

    x = tf.keras.layers.Concatenate()([x, inputLayer])

    return x

def generator(shape, numResNet):
    initialize = tf.keras.initializers.RandomNormal(stddev=0.02)
    InputLayer = tf.keras.layers.Input(shape=shape) 

    x = tf.keras.layers.Conv2D(64, (7,7), padding='same', kernel_initializer=initialize)(InputLayer)
    x = tfa.layers.InstanceNormalization(axis=-1)(x)
    x = tf.keras.activations.relu()(x)

    x = tf.keras.layers.Conv2D(128, (3,3), strides=(2,2), padding='same', kernel_initializer=initialize)(InputLayer)
    x = tfa.layers.InstanceNormalization(axis=-1)(x)
    x = tf.keras.activations.relu()(x)

    x = tf.keras.layers.Conv2D(256, (3,3), strides=(2,2), padding='same', kernel_initializer=initialize)(InputLayer)
    x = tfa.layers.InstanceNormalization(axis=-1)(x)
    x = tf.keras.activations.relu()(x)

    for _ in range(numResNet):
        x = resnet(256, x)
    
    x = tf.keras.layers.Conv2D(128, (3,3), strides=(2,2), padding='same', kernel_initializer=initialize)(InputLayer)
    x = tfa.layers.InstanceNormalization(axis=-1)(x)
    x = tf.keras.activations.relu()(x)

    x = tf.keras.layers.Conv2D(64, (3,3), strides=(2,2), padding='same', kernel_initializer=initialize)(InputLayer)
    x = tfa.layers.InstanceNormalization(axis=-1)(x)
    x = tf.keras.activations.relu()(x)

    x = tf.keras.layers.Conv2D(64, (7,7), padding='same', kernel_initializer=initialize)(InputLayer)
    x = tfa.layers.InstanceNormalization(axis=-1)(x)
    outputLayer = tf.keras.activations.tanh()(x)

    model = tf.keras.Model(InputLayer, outputLayer)

    return model