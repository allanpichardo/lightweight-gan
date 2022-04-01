import tensorflow as tf
import typing
import tensorflow_addons as tfa
from tensorflow import keras

if typing.TYPE_CHECKING:
    from keras.api._v2 import keras


class PixelwiseFeatureNormalization(keras.layers.Layer):

    def __init__(self, **kwargs):
        super(PixelwiseFeatureNormalization, self).__init__(**kwargs)

    # perform the operation
    def call(self, inputs, **kwargs):
        # calculate square pixel values
        values = inputs ** 2.0
        # calculate the mean pixel values
        mean_values = keras.backend.mean(values, axis=-1, keepdims=True)
        # ensure the mean is not zero
        mean_values += 1.0e-8
        # calculate the sqrt of the mean squared value (L2 norm)
        l2 = keras.backend.sqrt(mean_values)
        # normalize values by the l2 norm
        normalized = inputs / l2
        return normalized

    # define the output shape of the layer
    def compute_output_shape(self, input_shape):
        return input_shape
