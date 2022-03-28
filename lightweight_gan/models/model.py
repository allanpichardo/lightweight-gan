import os.path
import typing
from abc import ABC

import tensorflow as tf
from tensorflow import keras

from lightweight_gan.layers.image import Resize, StatelessCrop
from lightweight_gan.layers.residual_downsampling import ResidualDownsamplingBlock
from lightweight_gan.layers.simple_decoder import SimpleDecoder
from lightweight_gan.layers.skip_layer_excitation import SkipLayerExcitation
from lightweight_gan.layers.upsampling_convolution import UpsamplingConvolutionBlock

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

        self._random_flip = keras.layers.RandomFlip()
        self._random_zoom = keras.layers.RandomZoom(height_factor=(-0.3, 0.3), width_factor=(-0.3, 0.3))
        self._random_rotate = keras.layers.RandomRotation(factor=0.2)
        self._random_contrast = keras.layers.RandomContrast(factor=0.4)

        self._crop8x8 = StatelessCrop(8, 8)
        self._resize = Resize(128, 128)
        self._crop128x128 = StatelessCrop(128, 128)

        seed = tf.random.uniform([2], maxval=1024 * 1024, dtype=tf.int32)
        self._crop128x128.seed = seed
        self._crop8x8.seed = seed

        self._conv4x4_1 = None
        self._prelu1 = None
        self._conv4x4_2 = None
        self._batchnorm1 = None
        self._prelu2 = None

        self._downsampling_layers = []

        self.simple_decoder_i_part = None
        self.simple_decoder_i = None

        self._conv1x1 = None
        self._batchnorm2 = None
        self._prelu3 = None
        self._conv4x4_3 = None
        self._flatten = None
        self._dropout = None
        self._dense = None

    def build(self, input_shape):
        self._conv4x4_1 = keras.layers.Conv2D(32, (4, 4), strides=2, padding='same')
        self._prelu1 = keras.layers.PReLU(shared_axes=[1, 2])
        self._conv4x4_2 = keras.layers.Conv2D(32, (4, 4), strides=2, padding='same')
        self._batchnorm1 = keras.layers.LayerNormalization()
        self._prelu2 = keras.layers.PReLU(shared_axes=[1, 2])

        filters = 32
        for i in range(5):
            self._downsampling_layers.append(ResidualDownsamplingBlock(filters))
            filters = filters * 2

        self.simple_decoder_i_part = SimpleDecoder(256)
        self.simple_decoder_i = SimpleDecoder(512)

        self._conv1x1 = keras.layers.Conv2D(512, (1, 1))
        self._batchnorm2 = keras.layers.LayerNormalization()
        self._prelu3 = keras.layers.PReLU(shared_axes=[1, 2])
        self._conv4x4_3 = keras.layers.Conv2D(1, (4, 4))
        self._flatten = keras.layers.Flatten()
        self._dropout = keras.layers.Dropout(0.2)
        self._dense = keras.layers.Dense(1)

    def call(self, inputs, training=None, mask=None):
        #  Data augmentation
        x = self._random_flip(inputs)
        x = self._random_zoom(x)
        x = self._random_rotate(x)
        x = self._random_contrast(x)

        i = self._resize(x)
        i_part = self._crop128x128(x)

        x = self._conv4x4_1(inputs)
        x = self._prelu1(x)
        x = self._conv4x4_2(x)
        x = self._batchnorm1(x)
        x = self._prelu2(x)

        i_part_prime = None
        i_prime = None
        size = 256
        for layer in self._downsampling_layers:
            x = layer(x)
            size = size / 2

            if size == 16:
                i_part_prime = self._crop8x8(x)
                i_part_prime = self.simple_decoder_i_part(i_part_prime)
            if size == 8:
                i_prime = self.simple_decoder_i(x)

        x = self._conv1x1(x)
        x = self._batchnorm2(x)
        x = self._prelu3(x)
        x = self._conv4x4_3(x)
        x = self._flatten(x)
        x = self._dropout(x)
        x = self._dense(x)

        return x, i_part, i_part_prime, i, i_prime


class LightweightGan(keras.models.Model, ABC):

    def __init__(self, discriminator_extra_steps=5, gradient_penalty_weight=10.0):
        super(LightweightGan, self).__init__()

        self.generator = Generator()
        self.discriminator = Discriminator()

        self.d_steps = discriminator_extra_steps
        self.gp_weight = gradient_penalty_weight


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

    def autoencoder_loss(self, i, i_prime, i_part, i_part_prime):
        i_loss = keras.losses.mean_squared_error(
            i, i_prime
        )

        i_part_loss = keras.losses.mean_squared_error(
            i_part, i_part_prime
        )

        return tf.reduce_mean(i_loss), tf.reduce_mean(i_part_loss)

    # Define the loss functions for the discriminator,
    # which should be (fake_loss - real_loss).
    # We will add the gradient penalty later to this loss function.
    def discriminator_loss(self, real_img, fake_img):
        real_loss = tf.reduce_mean(real_img)
        fake_loss = tf.reduce_mean(fake_img)
        return fake_loss - real_loss

    # Define the loss functions for the generator.
    def generator_loss(self, fake_img):
        return -tf.reduce_mean(fake_img)

    def gradient_penalty(self, batch_size, real_images, fake_images):
        """ Calculates the gradient penalty.

        This loss is calculated on an interpolated image
        and added to the discriminator loss.
        """
        # Get the interpolated image
        alpha = tf.random.normal([batch_size, 1, 1, 1], 0.0, 1.0)
        diff = fake_images - real_images
        interpolated = real_images + alpha * diff

        with tf.GradientTape() as gp_tape:
            gp_tape.watch(interpolated)
            # 1. Get the discriminator output for this interpolated image.
            pred, _, _, _, _ = self.discriminator(interpolated, training=True)

        # 2. Calculate the gradients w.r.t to this interpolated image.
        grads = gp_tape.gradient(pred, [interpolated])[0]
        # 3. Calculate the norm of the gradients.
        norm = tf.sqrt(tf.reduce_sum(tf.square(grads), axis=[1, 2, 3]))
        gp = tf.reduce_mean((norm - 1.0) ** 2)
        return gp

    def adversarial_loss(self, real_logits, generated_logits):
        batch_size = tf.shape(real_logits)[0]
        # this is usually called the non-saturating GAN loss

        real_labels = tf.ones(shape=(batch_size, 1))
        generated_labels = tf.zeros(shape=(batch_size, 1))

        # the generator tries to produce images that the discriminator considers as real
        generator_loss = self.generator_loss(generated_logits)
        # generator_loss = keras.losses.binary_crossentropy(
        #     real_labels, generated_logits, from_logits=True
        # )
        # the discriminator tries to determine if images are real or generated
        discriminator_loss = self.discriminator_loss(real_logits, generated_logits)
        # discriminator_loss = keras.losses.hinge(
        #     tf.concat([real_labels, generated_labels], axis=0),
        #     tf.concat([real_logits, generated_logits], axis=0)
        # )
        # discriminator_loss = keras.losses.binary_crossentropy(
        #     tf.concat([real_labels, generated_labels], axis=0),
        #     tf.concat([real_logits, generated_logits], axis=0),
        #     from_logits=True,
        # )

        return generator_loss, discriminator_loss

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
        if isinstance(real_images, tuple):
            real_images = real_images[0]

        # Get the batch size
        batch_size = tf.shape(real_images)[0]

        # For each batch, we are going to perform the
        # following steps as laid out in the original paper:
        # 1. Train the generator and get the generator loss
        # 2. Train the discriminator and get the discriminator loss
        # 3. Calculate the gradient penalty
        # 4. Multiply this gradient penalty with a constant weight factor
        # 5. Add the gradient penalty to the discriminator loss
        # 6. Return the generator and discriminator losses as a loss dictionary

        # Train the discriminator first. The original paper recommends training
        # the discriminator for `x` more steps (typically 5) as compared to
        # one step of the generator. Here we will train it for 3 extra steps
        # as compared to 5 to reduce the training time.
        for i in range(self.d_steps):
            # Get the latent vector
            random_latent_vectors = tf.random.normal(
                shape=(batch_size, 256)
            )
            with tf.GradientTape() as tape:
                # Generate fake images from the latent vector
                fake_images = self.generator(random_latent_vectors, training=True)
                # Get the logits for the fake images
                fake_logits, _, _, _, _ = self.discriminator(fake_images, training=True)
                # Get the logits for the real images
                real_logits, i_part, i_part_prime, i, i_prime = self.discriminator(real_images, training=True)

                # Calculate the discriminator loss using the fake and real image logits
                d_cost = self.discriminator_loss(real_img=real_logits, fake_img=fake_logits)
                # Calculate the gradient penalty
                gp = self.gradient_penalty(batch_size, real_images, fake_images)
                # Add the gradient penalty to the original discriminator loss
                d_loss = d_cost + gp * self.gp_weight

                i_loss, i_part_loss = self.autoencoder_loss(i, i_prime, i_part, i_part_prime)
                decoder_loss = i_loss + i_part_loss / 2.0

            # Get the gradients w.r.t the discriminator loss
            d_gradient = tape.gradient([d_loss, decoder_loss], self.discriminator.trainable_variables)
            # Update the weights of the discriminator using the discriminator optimizer
            self.discriminator_optimizer.apply_gradients(
                zip(d_gradient, self.discriminator.trainable_variables)
            )

        # Train the generator
        # Get the latent vector
        random_latent_vectors = tf.random.normal(shape=(batch_size, 256))
        with tf.GradientTape() as tape:
            # Generate fake images using the generator
            generated_images = self.generator(random_latent_vectors, training=True)
            # Get the discriminator logits for fake images
            gen_img_logits, _, _, _, _ = self.discriminator(generated_images, training=True)
            # Calculate the generator loss
            g_loss = self.generator_loss(gen_img_logits)

        # Get the gradients w.r.t the generator loss
        gen_gradient = tape.gradient(g_loss, self.generator.trainable_variables)
        # Update the weights of the generator using the generator optimizer
        self.generator_optimizer.apply_gradients(
            zip(gen_gradient, self.generator.trainable_variables)
        )

        self.generator_loss_tracker.update_state(g_loss)
        self.discriminator_loss_tracker.update_state(d_loss)
        self.reconstruction_loss.update_state(decoder_loss)
        self.real_accuracy.update_state(1.0, LightweightGan.step(real_logits))
        self.generated_accuracy.update_state(0.0, LightweightGan.step(gen_img_logits))
        # self.augmentation_probability_tracker.update_state(self.augmenter.probability)

        return {m.name: m.result() for m in self.metrics[:-1]}

    # def train_step(self, real_images):
    #     batch_size = tf.shape(real_images)[0]
    #
    #     #  todo: implement augmentation
    #     # real_images = self.augmenter(real_images, training=True)
    #
    #     # use persistent gradient tape because gradients will be calculated twice
    #     with tf.GradientTape(persistent=True) as tape:
    #
    #         generated_images = self.generate(batch_size, training=True)
    #         # gradient is calculated through the image augmentation
    #         # generated_images = self.augmenter(generated_images, training=True)
    #
    #         # separate forward passes for the real and generated images, meaning
    #         # that batch normalization is applied separately
    #         real_logits, i_part, i_part_prime, i, i_prime = self.discriminator(real_images, training=True)
    #
    #         generated_logits, _, _, _, _ = self.discriminator(generated_images, training=True)
    #
    #         generator_loss, discriminator_loss = self.adversarial_loss(
    #             real_logits, generated_logits
    #         )
    #
    #         i_loss, i_part_loss = self.autoencoder_loss(i, i_prime, i_part, i_part_prime)
    #
    #     # calculate gradients and update weights
    #     generator_gradients = tape.gradient(
    #         generator_loss, self.generator.trainable_weights
    #     )
    #
    #     discriminator_gradients = tape.gradient(
    #         [discriminator_loss, i_loss, i_part_loss], self.discriminator.trainable_weights
    #     )
    #
    #     self.generator_optimizer.apply_gradients(
    #         zip(generator_gradients, self.generator.trainable_weights)
    #     )
    #
    #     self.discriminator_optimizer.apply_gradients(
    #         zip(discriminator_gradients, self.discriminator.trainable_weights)
    #     )
    #
    #     # update the augmentation probability based on the discriminator's performance
    #     # self.augmenter.update(real_logits)
    #
    #     self.generator_loss_tracker.update_state(generator_loss)
    #     self.discriminator_loss_tracker.update_state(discriminator_loss)
    #     self.reconstruction_loss.update_state(i_loss + i_part_loss)
    #     self.real_accuracy.update_state(1.0, LightweightGan.step(real_logits))
    #     self.generated_accuracy.update_state(0.0, LightweightGan.step(generated_logits))
    #     # self.augmentation_probability_tracker.update_state(self.augmenter.probability)
    #
    #     return {m.name: m.result() for m in self.metrics[:-1]}

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
    logits, real1, recon1, real2, recon2 = discriminator(img)
    discriminator.summary()

    assert logits.shape[1] == 1

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

    with tf.device('/cpu:0'):
        gan.fit(mock_dataset, epochs=5, callbacks=[keras.callbacks.LambdaCallback(on_epoch_end=gan.save_image_callback(
            os.path.join(os.path.dirname(__file__), 'generated')
        ))])