import tensorflow as tf


class Generative_function(tf.keras.layers.Layer):
    def __init__(self, noise_generator, num_samples, num_outputs, input_dim):

        """

        Parameters:
        -----------
        noise_generator :

        num_samples :

        num_outputs :

        input_dim :

        """
        super(Generative_function, self).__init__(dtype="float64")
        self.num_samples = num_samples
        self.num_outputs = num_outputs
        self.input_dim = input_dim
        self.noise_generator = noise_generator

    def call(self, inputs):
        """

        Parameters:
        -----------
        inputs :
        """
        raise NotImplementedError


class BayesianLinearNN(Generative_function):
    def __init__(self, noise_generator, num_samples, num_outputs=1, input_dim=32):

        initializer = tf.random_normal_initializer(mean=0, stddev=1.0)
        self.w_mean = tf.Variable(
            initial_value=0.01
            * initializer(shape=(input_dim, num_outputs), dtype="float64"),
            trainable=True,
            name="w_mean",
        )

        self.w_log_std = tf.Variable(
            initial_value=-5
            + initializer(shape=(input_dim, num_outputs), dtype="float64"),
            trainable=True,
            name="w_log_std",
        )

        self.b_mean = tf.Variable(
            initial_value=0.01 * initializer(shape=[num_outputs],
                                             dtype="float64"),
            trainable=True,
            name="b_mean",
        )

        self.b_log_std = tf.Variable(
            initial_value=-5 + initializer(shape=[num_outputs],
                                           dtype="float64"),
            trainable=True,
            name="b_std",
        )
        super().__init__(noise_generator, num_samples, num_outputs, input_dim)

    def call(self, inputs):

        z_w = self.noise_generator((self.num_samples, 1, self.num_outputs))
        z_b = self.noise_generator((self.num_samples, 1, self.num_outputs))

        w = z_w * tf.math.exp(self.w_log_std) + self.w_mean
        b = z_b * tf.math.exp(self.b_log_std) + self.b_mean

        return inputs @ w + b
