import os.path
from abc import ABC
import matplotlib.pyplot as plt
from lightweight_gan.layers.upsampling_convolution import UpsamplingConvolutionBlock
from lightweight_gan.layers.skip_layer_excitation import SkipLayerExcitation
from lightweight_gan.layers.image import Resize, StatelessCrop
from lightweight_gan.layers.residual_downsampling import ResidualDownsamplingBlock
from lightweight_gan.layers.simple_decoder import SimpleDecoder
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


class Discriminator(keras.models.Model, ABC):
    """
    Self-supervised discriminator. Returns 5 outputs:
    discriminator logits, 128x128 real cropped image, 128x128 decoded cropped image, 128x128 real scaled image, 128x128 decoded scaled image
    """

    def __init__(self):
        super(Discriminator, self).__init__()

        self._crop8x8 = StatelessCrop(8, 8)

        self._conv4x4_1 = None
        self._prelu1 = None
        self._conv4x4_2 = None
        self._batchnorm1 = None
        self._prelu2 = None

        self._downsampling_layers = []

        self._simple_decoder1 = None
        self._simple_decoder2 = None

        self._conv1x1 = None
        self._batchnorm2 = None
        self._prelu3 = None
        self._conv4x4_3 = None

    def build(self, input_shape):
        tf.config.run_functions_eagerly(True)

        self._conv4x4_1 = keras.layers.Conv2D(16, (4, 4), strides=2, padding='same')
        self._prelu1 = keras.layers.PReLU(shared_axes=[1, 2])
        self._conv4x4_2 = keras.layers.Conv2D(16, (4, 4), strides=2, padding='same')
        self._batchnorm1 = keras.layers.BatchNormalization()
        self._prelu2 = keras.layers.PReLU(shared_axes=[1, 2])

        filters = 16
        for i in range(5):
            self._downsampling_layers.append(ResidualDownsamplingBlock(filters))
            filters = filters * 2

        self._simple_decoder1 = SimpleDecoder(256)
        self._simple_decoder2 = SimpleDecoder(512)

        self._conv1x1 = keras.layers.Conv2D(512, (1, 1))
        self._batchnorm2 = keras.layers.BatchNormalization()
        self._prelu3 = keras.layers.PReLU(shared_axes=[1, 2])
        self._conv4x4_3 = keras.layers.Conv2D(1, (4, 4))

    def call(self, inputs, training=None, mask=None):
        x = self._conv4x4_1(inputs)
        x = self._prelu1(x)
        x = self._conv4x4_2(x)
        x = self._batchnorm1(x)
        x = self._prelu2(x)

        a = None
        b = None
        size = 256
        for layer in self._downsampling_layers:
            x = layer(x)
            size = size / 2

            if size == 16:
                a = self._crop8x8(x)
                a = self._simple_decoder1(a)
            if size == 8:
                b = self._simple_decoder2(x)

        x = self._conv1x1(x)
        x = self._batchnorm2(x)
        x = self._prelu3(x)
        x = self._conv4x4_3(x)

        return x, a, b


class LightweightGan(keras.models.Model, ABC):

    def __init__(self, ):
        super(LightweightGan, self).__init__()

        self.generator = Generator()
        self.discriminator = Discriminator()
        self._resize = Resize(128, 128)
        self._crop128x128 = StatelessCrop(128, 128)

    def compile(self, generator_optimizer, discriminator_optimizer, **kwargs):
        super(LightweightGan, self).compile(**kwargs)

        self.generator_optimizer = generator_optimizer
        self.discriminator_optimizer = discriminator_optimizer

        self.generator_loss_tracker = keras.metrics.Mean(name="g_loss")
        self.discriminator_loss_tracker = keras.metrics.Mean(name="d_loss")
        self.reconstruction_loss = keras.metrics.Mean(name="recon_loss")
        self.real_accuracy = keras.metrics.BinaryAccuracy(name="real_acc")
        self.generated_accuracy = keras.metrics.BinaryAccuracy(name="gen_acc")

    @property
    def metrics(self):
        return [
            self.generator_loss_tracker,
            self.discriminator_loss_tracker,
            self.reconstruction_loss,
            self.real_accuracy,
            self.generated_accuracy,
        ]

    def generate(self, batch_size, training):
        latent_samples = tf.random.normal(shape=[batch_size, 256])
        return self.generator(latent_samples, training)

    def autoencoder_loss(self, real_images, i_scaled_input, image_i, i_cropped_input, image_i_part):
        i_loss = keras.losses.mean_squared_error(
            i_scaled_input, image_i
        )

        i_part_loss = keras.losses.mean_squared_error(
            i_cropped_input, image_i_part
        )

        return tf.reduce_mean(i_loss) + tf.reduce_mean(i_part_loss)

    def adversarial_loss(self, real_logits, generated_logits):
        batch_size = tf.shape(real_logits)[0]
        # this is usually called the non-saturating GAN loss

        real_labels = tf.ones(shape=(batch_size, 5, 5, 1))
        generated_labels = tf.zeros(shape=(batch_size, 5, 5, 1))

        # the generator tries to produce images that the discriminator considers as real
        generator_loss = keras.losses.hinge(
            real_labels, generated_logits
        )
        # the discriminator tries to determine if images are real or generated
        discriminator_loss = keras.losses.hinge(
            tf.concat([real_labels, generated_labels], axis=0),
            tf.concat([real_logits, generated_logits], axis=0)
        )
        # discriminator_loss = keras.losses.binary_crossentropy(
        #     tf.concat([real_labels, generated_labels], axis=0),
        #     tf.concat([real_logits, generated_logits], axis=0),
        #     from_logits=True,
        # )

        return tf.reduce_mean(generator_loss), tf.reduce_mean(discriminator_loss)

    @staticmethod
    def step(values):
        # negative values -> 0.0, positive values -> 1.0
        return 0.5 * (1.0 + tf.sign(values))

    def call(self, inputs, training=None, mask=None):
        if training:
            return self.discriminator(self.generator(inputs))
        else:
            return self.generator(inputs)

    def train_step(self, real_images):
        batch_size = tf.shape(real_images)[0]
        i_scaled_input = self._resize(real_images)
        i_cropped_input = self._crop128x128(real_images)
        #  todo: implement augmentation
        # real_images = self.augmenter(real_images, training=True)

        # use persistent gradient tape because gradients will be calculated twice
        with tf.GradientTape(persistent=True) as tape:

            generated_images = self.generate(batch_size, training=True)
            # gradient is calculated through the image augmentation
            # generated_images = self.augmenter(generated_images, training=True)

            # separate forward passes for the real and generated images, meaning
            # that batch normalization is applied separately
            real_logits, i_crop, i_scaled = self.discriminator(real_images, training=True)

            generated_logits, _, _ = self.discriminator(generated_images, training=True)

            generator_loss, discriminator_loss = self.adversarial_loss(
                real_logits, generated_logits
            )

            reconstruction_loss = self.autoencoder_loss(real_images, i_scaled_input, i_scaled, i_cropped_input, i_crop)

        # calculate gradients and update weights
        generator_gradients = tape.gradient(
            generator_loss, self.generator.trainable_weights
        )

        discriminator_gradients = tape.gradient(
            [discriminator_loss, reconstruction_loss], self.discriminator.trainable_weights
        )
        self.generator_optimizer.apply_gradients(
            zip(generator_gradients, self.generator.trainable_weights)
        )
        self.discriminator_optimizer.apply_gradients(
            zip(discriminator_gradients, self.discriminator.trainable_weights)
        )

        # update the augmentation probability based on the discriminator's performance
        # self.augmenter.update(real_logits)

        self.generator_loss_tracker.update_state(generator_loss)
        self.discriminator_loss_tracker.update_state(discriminator_loss)
        self.reconstruction_loss.update_state(reconstruction_loss)
        self.real_accuracy.update_state(1.0, LightweightGan.step(real_logits))
        self.generated_accuracy.update_state(0.0, LightweightGan.step(generated_logits))
        # self.augmentation_probability_tracker.update_state(self.augmenter.probability)

        return {m.name: m.result() for m in self.metrics[:-1]}

    def plot_images(self, epoch=None, logs=None, num_rows=3, num_cols=6, interval=5):
        # plot random generated images for visual evaluation of generation quality
        if epoch is None or (epoch + 1) % interval == 0:
            num_images = num_rows * num_cols
            generated_images = self.generate(num_images, training=False)

            plt.figure(figsize=(num_cols * 2.0, num_rows * 2.0))
            for row in range(num_rows):
                for col in range(num_cols):
                    index = row * num_cols + col
                    plt.subplot(num_rows, num_cols, index + 1)
                    plt.imshow(generated_images[index])
                    plt.axis("off")
            plt.tight_layout()
            plt.show()
            plt.close()

    def save_image_callback(self, image_dir, interval=5, amount=10):
        def callback_function(epoch=None, logs=None):
            if epoch is None or (epoch + 1) % interval == 0:
                generated_images = self.generate(amount, training=False)
                for i, image in enumerate(generated_images):
                    jpg = tf.image.convert_image_dtype(image, tf.uint8, saturate=True)
                    jpg = tf.image.encode_jpeg(jpg)
                    tf.io.write_file(os.path.join(image_dir, str(epoch) if epoch is not None else '0', "image-{}.jpg".format(i)), jpg)

        return callback_function


if __name__ == '__main__':
    #  test generator
    z = tf.random.normal([1, 256])

    generator = Generator()
    img = generator(z)
    generator.summary()

    assert img.shape[1] == 1024
    assert img.shape[2] == 1024
    assert img.shape[3] == 3

    #  test discriminator

    discriminator = Discriminator()
    logits, recon1, recon2 = discriminator(img)
    discriminator.summary()

    assert logits.shape[1] == 5
    assert logits.shape[2] == 5
    assert logits.shape[3] == 1

    assert recon1.shape[1] == 128
    assert recon1.shape[2] == 128
    assert recon1.shape[3] == 3

    assert recon2.shape[1] == 128
    assert recon2.shape[2] == 128
    assert recon2.shape[3] == 3

    # test gan

    mock_dataset = tf.random.normal([8, 1024, 1024, 3], mean=0.5, stddev=0.5)

    gan = LightweightGan()
    gan.compile(
        generator_optimizer=keras.optimizers.Adam(),
        discriminator_optimizer=keras.optimizers.Adam(),
    )

    gan.fit(mock_dataset, epochs=5, callbacks=[keras.callbacks.LambdaCallback(on_epoch_end=gan.save_image_callback(
        os.path.join(os.path.dirname(__file__), 'generated')
    ))])
