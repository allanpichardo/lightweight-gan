from abc import ABC
from lightweight_gan.layers.upsampling_convolution import UpsamplingConvolutionBlock
from lightweight_gan.layers.skip_layer_excitation import SkipLayerExcitation
import tensorflow as tf
import typing

from tensorflow import keras

if typing.TYPE_CHECKING:
    from keras.api._v2 import keras


class Generator(keras.models.Model, ABC):
    """
    Takes a 256-dimensional 1-D tensor to generate a 1024x1024 image
    """

    def __init__(self):
        super(Generator, self).__init__()

        self._tanh = None
        self._conv3x3 = None
        self._prelu = None
        self._batchnorm = None
        self._upsample_blocks = []
        self._skip_layers = []
        self._conv_transpose = None

    def build(self, input_shape):
        self._conv_transpose = keras.layers.Conv2DTranspose(1024, (4, 4))
        self._batchnorm = keras.layers.BatchNormalization()
        self._prelu = keras.layers.PReLU(shared_axes=[1, 2])

        # goes from 512x4x4 -> 4x1024x1024 over 8 upsamples
        filters = 512
        for i in range(8):
            self._upsample_blocks.append(UpsamplingConvolutionBlock(filters))
            filters = int(filters / 2)

        for i in range(3):
            self._skip_layers.append(SkipLayerExcitation())

        self._conv3x3 = keras.layers.Conv2D(3, (3, 3), padding='same')
        self._tanh = tf.nn.tanh

    def call(self, inputs, training=None, mask=None):
        x = keras.layers.Reshape((1, 1, 256))(inputs)
        x = self._conv_transpose(x)
        x = self._batchnorm(x)
        x = self._prelu(x)

        eight = self._upsample_blocks[0](x)
        sixteen = self._upsample_blocks[1](eight)
        thirtytwo = self._upsample_blocks[2](sixteen)
        sixtyfour = self._upsample_blocks[3](thirtytwo)

        onetwoeight = self._upsample_blocks[4](sixtyfour)
        onetwoeight = self._skip_layers[0]([onetwoeight, eight])

        twofivesix = self._upsample_blocks[5](onetwoeight)
        twofivesix = self._skip_layers[1]([twofivesix, sixteen])

        fivetwelve = self._upsample_blocks[6](twofivesix)
        fivetwelve = self._skip_layers[2]([fivetwelve, thirtytwo])

        tentwofour = self._upsample_blocks[7](fivetwelve)

        x = self._conv3x3(tentwofour)
        x = self._tanh(x)

        return x


if __name__ == '__main__':
    #  test generator
    z = tf.random.normal([1, 256])

    generator = Generator()
    img = generator(z)

    assert img.shape[1] == 1024
    assert img.shape[2] == 1024
    assert img.shape[3] == 3


