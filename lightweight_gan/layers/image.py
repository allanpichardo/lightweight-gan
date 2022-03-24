import tensorflow as tf
import typing

from tensorflow import keras

if typing.TYPE_CHECKING:
    from keras.api._v2 import keras


class Resize(keras.layers.Layer):

    def __init__(self, height, width):
        super(Resize, self).__init__()

        self._channels = None
        self._width = width
        self._height = height

    def call(self, inputs, *args, **kwargs):
        return tf.image.resize(inputs, [self._height, self._width])


class StatelessCrop(keras.layers.Layer):

    def __init__(self, height, width, data_format='channels_last'):
        super(StatelessCrop, self).__init__()

        # self.seed = None
        self._channels = None
        self._batch_size = None
        self._data_format = data_format
        self._height = height
        self._width = width

        # self.update_seed()

    def build(self, input_shape):
        self._batch_size = input_shape[0]
        self._channels = input_shape[3] if self._data_format == 'channels_last' else input_shape[1]

    def call(self, inputs, *args, **kwargs):
        batch_size = tf.shape(inputs)[0]
        return tf.image.stateless_random_crop(inputs, [batch_size, self._height, self._width, self._channels], self.seed)

    # def update_seed(self):
    #     self.seed = tf.random.uniform([2], maxval=self._width * self._height, dtype=tf.int32)


if __name__ == '__main__':
    mock_image = tf.random.normal([1, 128, 128, 64])
    second_image = tf.identity(mock_image)

    assert tf.reduce_all(tf.math.equal(mock_image, second_image))

    layer = StatelessCrop(8, 8)
    crop1 = layer(mock_image)

    assert crop1.shape[1] == 8
    assert crop1.shape[2] == 8
    assert crop1.shape[3] == 64

    crop2 = layer(second_image)

    assert tf.reduce_all(tf.math.equal(crop1, crop2))

    layer.update_seed()

    crop1 = layer(mock_image)

    assert tf.reduce_all(tf.not_equal(crop1, crop2))

    crop2 = layer(second_image)

    assert tf.reduce_all(tf.math.equal(crop1, crop2))

    #  test resize

    resized = Resize(16, 16)(mock_image)

    assert resized.shape[1] == 16
    assert resized.shape[2] == 16
    assert resized.shape[3] == 64
