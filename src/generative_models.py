import tensorflow as tf
import numpy as np


class NoiseSampler:
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
    def __init__(self, noise_sampler, num_samples, num_outputs, input_dim):

        """
        Generates samples from a stochastic function using sampled
        noise values and input values.

        Suppose an input value is given in `x`, generates
        f_1,...,f_{num\_samples} using noise values and
        a function f(x, z)

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
        self.noise_sampler = noise_sampler
        self.num_samples = num_samples
        self.num_outputs = num_outputs
        self.input_dim = input_dim

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


class Linear(GenerativeFunction):
    def __init__(self, noise_sampler=None, num_samples=10, num_outputs=1, input_dim=32):
        """
        Parameters:
        -----------

        num_samples : int
                      Ammount of samples to generate in each call.

        num_outputs : int
                      Dimensionality of the function output.

        input_dim : int or array
                    Dimensionality of the input values `x`.


        """

        # Initialize tf variables
        self.w = tf.Variable(
            initial_value=1.0 * np.ones((input_dim, num_outputs)),
            trainable=True,
            name="w",
        )
        self.b = tf.Variable(
            initial_value=0.01 * np.ones((input_dim, num_outputs)),
            trainable=True,
            name="b",
        )

        super().__init__(None, num_samples, num_outputs, input_dim)

    def call(self, inputs):

        b = tf.expand_dims(self.b, 0)
        w = tf.expand_dims(self.w, 0)
        b = tf.tile(b, (self.num_samples, 1, 1))
        w = tf.tile(w, (self.num_samples, 1, 1))
        return inputs @ w + b


class BayesianLinearNN(GenerativeFunction):
    def __init__(self, noise_sampler, num_samples, num_outputs=1, input_dim=32):
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
            initial_value=0.01 * initializer(shape=[num_outputs], dtype="float64"),
            trainable=True,
            name="b_mean",
        )
        self.b_log_std = tf.Variable(
            initial_value=-5 + initializer(shape=[num_outputs], dtype="float64"),
            trainable=True,
            name="b_log_std",
        )

        super().__init__(noise_sampler, num_samples, num_outputs, input_dim)

    def call(self, inputs):

        z_w = self.noise_sampler((self.num_samples, self.input_dim, self.num_outputs))
        z_b = self.noise_sampler((self.num_samples, self.input_dim, self.num_outputs))

        w = z_w * tf.math.exp(self.w_log_std) + self.w_mean
        b = z_b * tf.math.exp(self.b_log_std) + self.b_mean

        return inputs @ w + b


class BayesianNN(GenerativeFunction):
    def __init__(
        self,
        noise_sampler,
        num_samples,
        structure,
        activation,
        num_outputs=1,
        input_dim=32,
    ):
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

        vars = []
        dims = [input_dim] + structure + [num_outputs]
        for _in, _out in zip(dims, dims[1:]):
            w_mean = tf.Variable(
                initial_value=0.01 * initializer(shape=(_in, _out), dtype="float64"),
                trainable=True,
                name="w_mean_" + str(_in) + "-" + str(_out),
            )
            w_log_std = tf.Variable(
                initial_value=-5 + initializer(shape=(_in, _out), dtype="float64"),
                trainable=True,
                name="w_log_std_" + str(_in) + "-" + str(_out),
            )
            b_mean = tf.Variable(
                initial_value=0.01 * initializer(shape=[_out], dtype="float64"),
                trainable=True,
                name="b_mean_" + str(_in) + "-" + str(_out),
            )
            b_log_std = tf.Variable(
                initial_value=-5 + initializer(shape=[_out], dtype="float64"),
                trainable=True,
                name="b_log_std_" + str(_in) + "-" + str(_out),
            )

            vars.append((w_mean, w_log_std, b_mean, b_log_std))

        # This has to be done this way or keras does not correctly add
        # the variables to the trainable_variables array
        self.vars = vars
        self.activation = activation
        super().__init__(noise_sampler, num_samples, num_outputs, input_dim)

    def call(self, inputs):

        # Input has shape (N, D), we are replicating it self.num_samples
        # times in the first dimension (S, N, D)
        x = tf.expand_dims(inputs, 0)
        x = tf.tile(x, (self.num_samples, 1, 1))

        for (w_m, w_log_std, b_m, b_log_std) in self.vars[:-1]:
            # Get noise
            z_w = self.noise_sampler((self.num_samples, *w_log_std.shape))
            z_b = self.noise_sampler((self.num_samples, 1, *b_log_std.shape))

            # Compute Gaussian variables
            w = z_w * tf.math.exp(w_log_std) + w_m
            b = z_b * tf.math.exp(b_log_std) + b_m

            x = self.activation(x @ w + b)

        w_m, w_log_std, b_m, b_log_std = self.vars[-1]
        z_w = self.noise_sampler((self.num_samples, *w_log_std.shape))
        z_b = self.noise_sampler((self.num_samples, 1, *b_log_std.shape))

        w = z_w * tf.math.exp(w_log_std) + w_m
        b = z_b * tf.math.exp(b_log_std) + b_m
        return x @ w + b


def get_bn(structure, act):
    def bn(noise_sampler, num_samples, input_dim, num_outputs):
        return BayesianNN(
            noise_sampler=noise_sampler,
            num_samples=num_samples,
            input_dim=input_dim,
            structure=structure,
            activation=act,
            num_outputs=num_outputs,
        )

    return bn
