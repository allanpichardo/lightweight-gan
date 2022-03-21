import tensorflow as tf
import typing

from tensorflow import keras

if typing.TYPE_CHECKING:
    from keras.api._v2 import keras


class AdaptivePooling2D(keras.layers.Layer):
    def __init__(self, output_height, output_width, data_format='channels_last'):
        super(AdaptivePooling2D, self).__init__()

        self._pool_size = None
        self._input_width = None
        self._input_height = None
        self._avg_pooling = None

        self._output_height = output_height
        self._output_width = output_width
        self._data_format = data_format

    def build(self, input_shape):
        self._input_height = input_shape[1] if self._data_format == 'channels_last' else input_shape[2]
        self._input_width = input_shape[2] if self._data_format == 'channels_last' else input_shape[3]
        self._pool_size = (int(self._input_height / self._output_height), int(self._input_width / self._output_width))

        self._avg_pooling = keras.layers.AveragePooling2D(self._pool_size, padding='same',
                                                          data_format=self._data_format)

    def call(self, inputs, *args, **kwargs):
        return self._avg_pooling(inputs)


class SkipLayerExcitation(keras.layers.Layer):
    """
    Skip Layer Excitation block as published in https://arxiv.org/abs/2101.04775

    Takes a list of feature maps of size 2
    """

    def __init__(self, data_format='channels_last'):
        super(SkipLayerExcitation, self).__init__()

        self._mult = None
        self._sigmoid = None
        self._convolution1x1 = None
        self._prelu = None
        self._convolution4x4 = None
        self._adaptive_pooling = None

        self._data_format = data_format

    def build(self, input_shape):
        large_shape = input_shape[0]
        large_filters = large_shape[3] if self._data_format == 'channels_last' else large_shape[1]

        self._adaptive_pooling = AdaptivePooling2D(4, 4, data_format=self._data_format)
        self._convolution4x4 = keras.layers.Conv2D(large_filters, (4, 4), data_format=self._data_format)
        self._prelu = keras.layers.PReLU(shared_axes=[1, 2])
        self._convolution1x1 = keras.layers.Conv2D(large_filters, (1, 1), data_format=self._data_format)
        self._sigmoid = tf.nn.sigmoid
        self._mult = keras.layers.Multiply()

    def call(self, inputs, *args, **kwargs):
        """
        Call the block on two feature maps
        :param inputs: must be a list of len == 2. Index 0 should be the large image, Index 1 is the small
        :param args:
        :param kwargs:
        :return:
        """
        x = self._adaptive_pooling(inputs[1])
        x = self._convolution4x4(x)
        x = self._prelu(x)
        x = self._convolution1x1(x)
        x = self._sigmoid(x)
        x = self._mult([inputs[0], x])
        return x


if __name__ == '__main__':
    mock_image_large = tf.random.normal([1, 128, 128, 64])
    mock_image_small = tf.random.normal([1, 8, 8, 512])

    # ----- Test adaptive pooling -----
    adaptive_pooling = AdaptivePooling2D(4, 4)
    resized = adaptive_pooling(mock_image_large)

    assert resized.shape[0] == 1
    assert resized.shape[1] == 4
    assert resized.shape[2] == 4
    assert resized.shape[3] == 64

    # ----- Test skip layer excitation -----

    sle = SkipLayerExcitation()
    feature_map = sle([mock_image_large, mock_image_small])

    assert len(feature_map.shape) == 4
    assert feature_map.shape[0] == 1
    assert feature_map.shape[1] == 128
    assert feature_map.shape[2] == 128
    assert feature_map.shape[3] == 64
