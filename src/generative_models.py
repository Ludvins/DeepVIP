import tensorflow as tf


class NoiseSampler():
    def __init__(self, function):
        """
        Generates noise samples using the two provided arguments.

        Parameters:
        -----------
        function : callable
                   Returns as many noise samples as indicated.

        size : int or array
               Indicates the size/shape of the noise values
               to be sampled
        """
        self.function = function

    def call(self, size):
        """
        Returns sampled noise values using the function and shape
        provided in the constructor. Alternatively, an specific
        shape can be given.

        Parameters:
        -----------

        size : int or array
               Indicates the size/shape of the noise values
               to be sampled. None by default, in this case
               self.size is used.

        Returns:
        --------

        samples : int or array
                  Contains the sampled values
        """
        return self.function(size)


class GenerativeFunction(tf.keras.layers.Layer):
    def __init__(self, num_samples, num_outputs, input_dim):

        """
        Generates samples from a stochastic function using sampled
        noise values and input values.

        Suppose an input value is given in `x`, generates
        \(f_1,\dots,f_{num\_samples}\) using noise values and
        a function
        \[
            f(x, z)
        \]

        Parameters:
        -----------

        num_samples : int
                      Ammount of samples to generate in each call.

        num_outputs : int
                      Dimensionality of the function output.

        input_dim : int or array
                    Dimensionality of the input values `x`.

        """
        super(GenerativeFunction, self).__init__(dtype="float64")
        self.num_samples = num_samples
        self.num_outputs = num_outputs
        self.input_dim = input_dim

    def get_noise_shape(self):
        """
        Returns the shape the noise must have for each given input.
        """
        raise NotImplementedError

    def call(self, inputs, noise):
        """
        Generates the function samples.

        Parameters:
        -----------
        inputs : tf.tensor
                 Contains the input values from which to generate
                 num_samples function values for each of them.

        noise : tf.tensor
                Contains the noise values from which to generate
                the function values for each input.
        """
        raise NotImplementedError


class BayesianLinearNN(GenerativeFunction):
    def __init__(self, num_samples, num_outputs=1, input_dim=32):
        """
        Defines a Bayesian Neural Network with 1 layer as a generative model.
        The defined model is the following:
        \[
            w \sim \mathcal{N}(w_mean, \exp(2*w_log_std))
        \]
        \[
            b \sim \mathcal{N}(b_mean, \exp(2*b_log_std))
        \]
        where given an input location `x` and two noise values z_w, z_b,
        a function sample is computed as:
        \[
            f(x, z) = (w_mean + z_w * \exp(w_log_std)) x + (b_mean + z_b * \exp(b_log_std))
        \]

        Parameters:
        -----------

        num_samples : int
                      Ammount of samples to generate in each call.

        num_outputs : int
                      Dimensionality of the function output.

        input_dim : int or array
                    Dimensionality of the input values `x`.


        """
        initializer = tf.random_normal_initializer(mean=0, stddev=1.0)

        # Initialize tf variables
        self.w_mean = tf.Variable(
            initial_value=0.01* initializer(shape=(input_dim, num_outputs),
                                            dtype="float64"),
            trainable=True,
            name="w_mean",
        )
        self.w_log_std = tf.Variable(
            initial_value=-5 + initializer(shape=(input_dim, num_outputs),
                                           dtype="float64"),
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
            name="b_log_std",
        )

        super().__init__(num_samples, num_outputs, input_dim)

    def get_noise_shape(self):
        return 2*self.num_outputs * self.num_samples

    def call(self, inputs, noise):

        # (num_samples, num_outputs)
        noise = tf.reshape(noise, [2, self.num_samples, 1, self.num_outputs])
        # Split noise values
        z_w = noise[0]
        z_b = noise[1]
        # z_w = self.noise_generator((self.num_samples, 1, self.num_outputs))
        # z_b = self.noise_generator((self.num_samples, 1, self.num_outputs))

        w = z_w * tf.math.exp(self.w_log_std) + self.w_mean
        b = z_b * tf.math.exp(self.b_log_std) + self.b_mean

        return inputs @ w + b


class BayesianNN_2Layers(GenerativeFunction):
    def __init__(self, num_samples, num_outputs=1, input_dim=32):
        """
        Defines a Bayesian Neural Network

        Parameters:
        -----------

        num_samples : int
                      Ammount of samples to generate in each call.

        num_outputs : int
                      Dimensionality of the function output.

        input_dim : int or array
                    Dimensionality of the input values `x`.


        """
        initializer = tf.random_normal_initializer(mean=0, stddev=1.0)

        # Initialize tf variables
        self.w_mean = tf.Variable(
            initial_value=0.01 * initializer(shape=(input_dim, 10),
                                            dtype="float64"),
            trainable=True,
            name="w_mean",
        )
        self.w_log_std = tf.Variable(
            initial_value=-5 + initializer(shape=(input_dim, 10),
                                           dtype="float64"),
            trainable=True,
            name="w_log_std",
        )
        self.b_mean = tf.Variable(
            initial_value=0.01 * initializer(shape=[10],
                                             dtype="float64"),
            trainable=True,
            name="b_mean",
        )
        self.b_log_std = tf.Variable(
            initial_value=-5 + initializer(shape=[10],
                                           dtype="float64"),
            trainable=True,
            name="b_log_std",
        )


        # Initialize tf variables
        self.w_mean_2 = tf.Variable(
            initial_value=0.01* initializer(shape=(10, num_outputs),
                                            dtype="float64"),
            trainable=True,
            name="w_mean_2",
        )
        self.w_log_std_2 = tf.Variable(
            initial_value=-5 + initializer(shape=(10, num_outputs),
                                           dtype="float64"),
            trainable=True,
            name="w_log_std_2",
        )
        self.b_mean_2 = tf.Variable(
            initial_value=0.01 * initializer(shape=[num_outputs],
                                             dtype="float64"),
            trainable=True,
            name="b_mean_2",
        )
        self.b_log_std_2 = tf.Variable(
            initial_value=-5 + initializer(shape=[num_outputs],
                                           dtype="float64"),
            trainable=True,
            name="b_log_std_2",
        )

        super().__init__(num_samples, num_outputs, input_dim)

    def get_noise_shape(self):
        return 4*self.num_outputs * self.num_samples

    def call(self, inputs, noise):

        # (num_samples, num_outputs)
        noise = tf.reshape(noise, [4, self.num_samples, 1, self.num_outputs])
        # Split noise values
        z_w_1 = noise[0]
        z_b_1 = noise[1]
        z_w_2 = noise[2]
        z_b_2 = noise[3]

        w = z_w_1 * tf.math.exp(self.w_log_std) + self.w_mean
        b = z_b_1 * tf.math.exp(self.b_log_std) + self.b_mean

        x = tf.math.tanh(inputs @ w + b)
        w = z_w_2 * tf.math.exp(self.w_log_std_2) + self.w_mean_2
        b = z_b_2 * tf.math.exp(self.b_log_std_2) + self.b_mean_2

        return x @ w + b

