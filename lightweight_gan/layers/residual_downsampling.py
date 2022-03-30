import tensorflow as tf
import typing
import tensorflow_addons as tfa

from tensorflow import keras

if typing.TYPE_CHECKING:
    from keras.api._v2 import keras


class ResidualDownsamplingBlock(keras.layers.Layer):

    def __init__(self, filters, data_format='channels_last'):
        super(ResidualDownsamplingBlock, self).__init__()

        self._filters = filters
        self._data_format = data_format

        self._convolution4x4 = None
        self._batch_norm1 = None
        self._prelu1 = None
        self._convolution3x3 = None
        self._batch_norm2 = None
        self._prelu2 = None

        self._average_pool = None
        self._convolution1x1 = None
        self._batch_norm3 = None
        self._prelu3 = None

    def build(self, input_shape):
        self._convolution4x4 = tfa.layers.SpectralNormalization(keras.layers.Conv2D(self._filters, (4, 4), strides=2, padding='same',
                                                   data_format=self._data_format))
        self._batch_norm1 = keras.layers.BatchNormalization()
        self._prelu1 = keras.layers.PReLU(shared_axes=[1, 2])
        self._convolution3x3 = tfa.layers.SpectralNormalization(keras.layers.Conv2D(self._filters, (3, 3), padding='same', data_format=self._data_format))
        self._batch_norm2 = keras.layers.BatchNormalization()
        self._prelu2 = keras.layers.PReLU(shared_axes=[1, 2])

        self._average_pool = keras.layers.AveragePooling2D()
        self._convolution1x1 = tfa.layers.SpectralNormalization(keras.layers.Conv2D(self._filters, (1, 1), data_format=self._data_format))
        self._batch_norm3 = keras.layers.BatchNormalization()
        self._prelu3 = keras.layers.PReLU(shared_axes=[1, 2])

    def call(self, inputs, *args, **kwargs):
        x = self._convolution4x4(inputs)
        x = self._batch_norm1(x)
        x = self._prelu1(x)
        x = self._convolution3x3(x)
        x = self._batch_norm2(x)
        x = self._prelu2(x)

        y = self._average_pool(inputs)
        y = self._convolution1x1(y)
        y = self._batch_norm3(y)
        y = self._prelu3(y)

        return tf.add(x, y)


if __name__ == '__main__':
    mock_image = tf.random.normal([1, 128, 128, 64])

    output = ResidualDownsamplingBlock(128)(mock_image)

    assert output.shape[0] == 1
    assert output.shape[1] == 64
    assert output.shape[2] == 64
    assert output.shape[3] == 128
