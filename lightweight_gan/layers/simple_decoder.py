from abc import ABC

from lightweight_gan.layers.upsampling_convolution import UpsamplingConvolutionBlock
import tensorflow as tf
import typing

from tensorflow import keras

if typing.TYPE_CHECKING:
    from keras.api._v2 import keras


class SimpleDecoder(keras.layers.Layer):

    def __init__(self, filters, image_channels=3, name="simple_decoder"):
        super(SimpleDecoder, self).__init__(name=name)

        self._layer1 = UpsamplingConvolutionBlock(filters)
        self._layer2 = UpsamplingConvolutionBlock(int(filters / 2))
        self._layer3 = UpsamplingConvolutionBlock(int(filters / 4))
        self._layer4 = UpsamplingConvolutionBlock(image_channels)

    def call(self, inputs, training=None, mask=None):
        x = self._layer1(inputs)
        x = self._layer2(x)
        x = self._layer3(x)
        x = self._layer4(x)
        return x


if __name__ == '__main__':
    mock_image = tf.random.normal([1, 8, 8, 64])

    layer = SimpleDecoder(64)
    out = layer(mock_image)

    assert out.shape[1] == 128
    assert out.shape[2] == 128
    assert out.shape[3] == 3
