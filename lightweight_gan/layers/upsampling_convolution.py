import tensorflow as tf
import typing
import tensorflow_addons as tfa
from tensorflow import keras

from .pixelwise_feature_normalization import PixelwiseFeatureNormalization

if typing.TYPE_CHECKING:
    from keras.api._v2 import keras


class UpsamplingConvolutionBlock(keras.layers.Layer):

    def __init__(self, filters, data_format='channels_last'):
        super(UpsamplingConvolutionBlock, self).__init__()

        self._filters = filters
        self._data_format = data_format

        self._upsampling = None
        self._convolution3x3 = None
        self._batch_norm = None
        self._prelu = None

    def build(self, input_shape):
        self._upsampling = keras.layers.UpSampling2D()
        self._convolution3x3 = tfa.layers.SpectralNormalization(keras.layers.Conv2D(self._filters, (3, 3), padding='same', data_format=self._data_format, kernel_initializer='he_normal'))
        self._batch_norm = keras.layers.BatchNormalization()
        self._prelu = keras.layers.PReLU(shared_axes=[1, 2])

    def call(self, inputs, *args, **kwargs):
        x = self._upsampling(inputs)
        x = self._convolution3x3(x)
        x = self._batch_norm(x)
        x = self._prelu(x)
        return x


if __name__ == '__main__':
    mock_image = tf.random.normal([1, 128, 128, 64])

    output = UpsamplingConvolutionBlock(128)(mock_image)

    assert output.shape[0] == 1
    assert output.shape[1] == 256
    assert output.shape[2] == 256
    assert output.shape[3] == 128
