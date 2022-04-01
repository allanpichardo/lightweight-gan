import tensorflow as tf
import typing
import tensorflow_addons as tfa
from tensorflow import keras

if typing.TYPE_CHECKING:
    from keras.api._v2 import keras


class PixelwiseFeatureNormalization(keras.layers.Layer):

    def __init__(self):
        super(PixelwiseFeatureNormalization, self).__init__()

    def call(self, inputs, *args, **kwargs):
        normalization_constant = keras.backend.sqrt(
            keras.backend.mean(inputs ** 2, axis=-1, keepdims=True) + 1.0e-8
        )
        return inputs / normalization_constant
