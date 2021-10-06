import tensorflow as tf


class Linear(tf.keras.layers.Layer):
    def __init__(self, noise_generator, num_samples, num_outputs=1, input_dim=32):
        """

        """
        super(Linear, self).__init__(dtype="float64")
        self.noise_generator = noise_generator

        w_init = tf.random_normal_initializer()
        self.w = tf.Variable(
            initial_value=w_init(shape=(input_dim, num_outputs), dtype="float64"),
            trainable=True,
            name="Generative function weights",
        )

        b_init = tf.zeros_initializer()
        self.b = tf.Variable(
            initial_value=b_init(shape=(num_outputs), dtype="float64"),
            trainable=True,
            name="Generative function bias",
        )

        self.num_samples = num_samples

        self.z_w = self.noise_generator((num_samples, 1, num_outputs))
        self.z_b = self.noise_generator((num_samples, 1, num_outputs))

    def call(self, inputs, num_samples):
        wx = tf.matmul(inputs, self.w)
        wx = tf.expand_dims(wx, 0)
        wx = tf.tile(wx, [self.num_samples, 1, 1])
        wx = self.z_w * wx
        b = self.z_b * self.b
        return wx + b
