import argparse
from tensorflow.python.keras.engine.input_layer import InputLayer
from tensorflow.python.ops.init_ops_v2 import Initializer
import tensorflow_addons as tfa
import tensorflow as tf


def discriminator(shape):
    initalize = tf.keras.initializers.RandomNormal(stddev=0.02)
    InputLayer = tf.keras.layers.Input(shape=shape)

    x = tf.keras.layers.Conv2D(64, (4,4), strides=(2,2), padding='same', kernel_initializer=initalize)(InputLayer)
    x = tf.keras.activations.LeakyReLU(0.2)(x)

    x = tf.keras.layers.Conv2D(128, (4,4), strides=(2,2), padding='same', kernel_initializer=initalize)(x)
    x = tfa.layers.InstanceNormalization(axis=-1)(x)
    x = tf.keras.activations.LeakyReLU(0.2)(x)

    x = tf.keras.layers.Conv2D(256, (4,4), strides=(2,2), padding='same', kernel_initializer=initalize)(x)
    x = tfa.layers.InstanceNormalization(axis=-1)(x)
    x = tf.keras.activations.LeakyReLU(0.2)(x)

    x = tf.keras.layers.Conv2D(512, (4,4), strides=(2,2), padding='same', kernel_initializer=initalize)(x)
    x = tfa.layers.InstanceNormalization(axis=-1)(x)
    x = tf.keras.activations.LeakyReLU(0.2)(x)

    x = tf.keras.layers.Conv2D(512, (4,4), strides=(2,2), padding='same', kernel_initializer=initalize)(x)
    x = tfa.layers.InstanceNormalization(axis=-1)(x)
    x = tf.keras.activations.LeakyReLU(0.2)(x)

    OutputLayer = tf.keras.layers.Conv2D(1, (4,4), padding='same', kernel_initializer=initalize)(x)

    model = tf.keras.Model(InputLayer, OutputLayer)
    model.compile(loss=tf.keras.losses.MeanSquaredError, optimizer=tf.keras.Optimizers.Adam(lr=0.0002))

    return model

