import tensorflow as tf
import numpy as np
from numpy.random import default_rng


class NoiseSampler:
    def __init__(self, seed):
        """
        Generates noise samples.

        Parameters:
        -----------
        seed : int
               Integer value used to generate reproducible results.

        """
        self.seed = seed

    def call(self):
        """
        Returns sampled noise values.
        """
        raise NotImplementedError


class GaussianSampler(NoiseSampler):
    def __init__(self, seed):
        """
        Generates noise samples from a Standar Gaussian distribution N(0, 1).

        Parameters:
        -----------
        seed : int
               Integer value used to generate reproducible results.

        """
        super().__init__(seed)
        self.rng = tf.random.Generator.from_seed(self.seed)

    def __call__(self, size):
        """
        Returns sampled noise values os the given size or shape.

        Parameters:
        -----------
        size : int or np.darray
               Indicates the desired shape/size of the sample to generate.

        Returns:
        --------
        samples : np.darray of shape (size)
                  A sample from a Gaussian distribution N(0, I).

        """
        return tf.random.normal(size, dtype=tf.float64)


class GenerativeFunction(tf.Module):
    def __init__(
        self,
        noise_sampler,
        num_samples,
        num_outputs,
        input_dim,
        trainable,
        seed,
    ):
        """
        Generates samples from a stochastic function using sampled
        noise values and input values.

        Suppose an input value is given in `x`, generates
        f_1,...,f_{num\_samples} using noise values and
        a function f(x, z).

        Parameters:
        -----------
        noise_sampler : NoiseSampler
                        Instance of NoiseSampler used to generate the noise
                        values.

        num_samples : int
                      Amount of samples to generate in each call.

        num_outputs : int
                      Dimensionality of the function output.

        input_dim : int or array
                    Dimensionality of the input values `x`.

        trainable : boolean
                    Determines wether the variables are trainable or not.

        seed : int
               Integer value used to generate reproducible results.
        """
        super(GenerativeFunction, self).__init__()
        self.noise_sampler = noise_sampler
        self.num_samples = num_samples
        self.num_outputs = num_outputs
        self.input_dim = input_dim
        self.seed = seed
        self.trainable = trainable

    def __call__(self):
        """
        Generates the function samples.
        """
        raise NotImplementedError


class Linear(GenerativeFunction):
    def __init__(self,
                 num_samples=10,
                 num_outputs=1,
                 input_dim=32,
                 trainable=True,
                 *args):
        """
        Generates samples from a linear deterministic function.

        Parameters:
        -----------

        num_samples : int
                      Amount of samples to generate in each call.

        num_outputs : int
                      Dimensionality of the function output.

        input_dim : int or array
                    Dimensionality of the input values `x`.

        trainable : boolean
                    Determines wether the variables are trainable or not.

        """

        # Initialize tf variables
        self.w = tf.Variable(
            initial_value=1.0 * np.ones((input_dim, num_outputs)),
            trainable=trainable,
            name="w",
        )
        self.b = tf.Variable(
            initial_value=0.01 * np.ones((input_dim, num_outputs)),
            trainable=trainable,
            name="b",
        )

        super().__init__(None, num_samples, num_outputs, input_dim, None)

    def __call__(self, inputs):
        """
        Generates the output of the Linear transformation as

        y = w^T x + b

        """
        b = tf.expand_dims(self.b, 0)
        w = tf.expand_dims(self.w, 0)
        b = tf.tile(b, (self.num_samples, 1, 1))
        w = tf.tile(w, (self.num_samples, 1, 1))
        return inputs @ w + b


class BayesianLinearNN(GenerativeFunction):
    def __init__(
        self,
        noise_sampler,
        num_samples,
        num_outputs=1,
        input_dim=32,
        trainable=True,
        seed=0,
    ):
        """
        Defines a Bayesian Neural Network with 1 layer as a generative model.
        The defined model is the following:

        w ~ N(w_mean, exp(2*w_log_std))
        b ~ N(b_mean, exp(2*b_log_std))

        where given an input location x and two noise values z_w, z_b,
        a function sample is computed as:

        f(x, z) = (w_mean + z_w * exp(w_log_std)) x
                + (b_mean + z_b * exp(b_log_std))

        Parameters:
        -----------
        noise_sampler : NoiseSampler
                        Instance of NoiseSampler used to generate the noise
                        values.
        num_samples : int
                      Amount of samples to generate in each call.
        num_outputs : int
                      Dimensionality of the function output.
        input_dim : int or array
                    Dimensionality of the input values x.
        seed : int
               Integer value used to generate reproducible results.
        """
        initializer = tf.random_normal_initializer(mean=0,
                                                   stddev=1.0,
                                                   seed=seed)

        # Initialize tf variables
        self.w_mean = tf.Variable(
            initial_value=0.01 *
            initializer(shape=(input_dim, num_outputs), dtype="float64"),
            trainable=trainable,
            name="w_mean",
        )
        self.w_log_std = tf.Variable(
            initial_value=-5 +
            initializer(shape=(input_dim, num_outputs), dtype="float64"),
            trainable=trainable,
            name="w_log_std",
        )
        self.b_mean = tf.Variable(
            initial_value=0.01 *
            initializer(shape=[num_outputs], dtype="float64"),
            trainable=trainable,
            name="b_mean",
        )
        self.b_log_std = tf.Variable(
            initial_value=-5 +
            initializer(shape=[num_outputs], dtype="float64"),
            trainable=trainable,
            name="b_log_std",
        )

        super().__init__(noise_sampler, num_samples, num_outputs, input_dim,
                         trainable, seed)

    def __call__(self, inputs):
        """
        Generates the output of the stochastic function

        f(x, z) = (w_mean + z_w * \exp(w_log_std)) x
                + (b_mean + z_b * \exp(b_log_std)).
        """

        x = tf.expand_dims(inputs, 1)
        x = tf.tile(x, (1, self.num_samples, 1))

        z_w = self.noise_sampler(
            (self.num_samples, self.input_dim, self.num_outputs))
        z_b = self.noise_sampler((self.num_samples, self.num_outputs))

        w = z_w * tf.math.exp(self.w_log_std) + self.w_mean
        b = z_b * tf.math.exp(self.b_log_std) + self.b_mean

        return tf.einsum("nsi,sio->nso", x, w) + b


class BayesianNN(GenerativeFunction):
    def __init__(
        self,
        noise_sampler,
        num_samples,
        structure,
        activation,
        num_outputs=1,
        input_dim=1,
        trainable=True,
        w_mean_prior=0.0,
        w_std_prior=-4.6,
        b_mean_prior=0.01,
        b_std_prior=0.0,
        seed=0,
    ):
        """
        Defines a Bayesian Neural Network.

        Parameters:
        -----------
        noise_sampler : NoiseSampler
                        Instance of NoiseSampler used to generate the noise
                        values.
        num_samples : int
                      Amount of samples to generate in each call.
        structure : array-like
                    Contains the inner dimensions of the Bayesian Neural
                    network. For example, [10, 10] symbolizes a Bayesian
                    network with 2 inner layers of width 10.
        activation : function
                     Activation function to use between inner layers.
        num_outputs : int
                      Dimensionality of the function output.
        input_dim : int or array
                    Dimensionality of the input values `x`.
        trainable : boolean
                    Determines wether the variables are trainable or not.
        seed : int
               Integer value used to generate reproducible results.
        """
        self.initializer = tf.random_normal_initializer(mean=0,
                                                        stddev=1.0,
                                                        seed=seed)

        self.structure = structure
        self.w_mean_prior = w_mean_prior
        self.w_std_prior = w_std_prior
        self.b_mean_prior = b_mean_prior
        self.b_std_prior = b_std_prior
        self.trainable = trainable
        self.output_dim = num_outputs

        self.activation = activation
        super().__init__(noise_sampler, num_samples, num_outputs, input_dim,
                         trainable, seed)
        dims = [input_dim] + self.structure + [self.output_dim]

        weights = []

        for _in, _out in zip(dims, dims[1:]):
            w_mean = tf.Variable(
                initial_value=self.w_mean_prior +
                0.01 * self.initializer(shape=(_in, _out), dtype="float64"),
                trainable=self.trainable,
                name="w_mean_" + str(_in) + "-" + str(_out),
            )
            w_log_std = tf.Variable(
                initial_value=self.w_std_prior + 4.6 +
                0.0 * self.initializer(shape=(_in, _out), dtype="float64"),
                trainable=self.trainable,
                name="w_log_std_" + str(_in) + "-" + str(_out),
            )
            b_mean = tf.Variable(
                initial_value=self.b_mean_prior +
                0.1 * self.initializer(shape=[_out], dtype="float64"),
                trainable=self.trainable,
                name="b_mean_" + str(_in) + "-" + str(_out),
            )
            b_log_std = tf.Variable(
                initial_value=self.b_std_prior +
                0.1 * self.initializer(shape=[_out], dtype="float64"),
                trainable=self.trainable,
                name="b_log_std_" + str(_in) + "-" + str(_out),
            )

            weights.append((w_mean, w_log_std, b_mean, b_log_std))

        # This has to be done this way or keras does not correctly add
        # the variables to the trainable_variables array
        self.vars = weights

    def __call__(self, inputs):
        """
        Computes the output of the Bayesian neural network given the input
        values.
        """

        # inputs shape (S, M, N, D_in)
        x = inputs
        for (w_m, w_log_std, b_m, b_log_std) in self.vars[:-1]:
            # Get noise

            # (S, M, D_out, D_in)
            z_w = self.noise_sampler(
                tf.concat(
                    [tf.shape(x)[:-2], tf.shape(w_log_std)], 0))
            # (S, M, 1, D_out)
            z_b = self.noise_sampler(
                tf.concat([tf.shape(x)[:-2], [1],
                           tf.shape(b_log_std)], 0))

            # Compute Gaussian samples
            w = z_w * tf.math.exp(w_log_std) + w_m
            b = z_b * tf.math.exp(b_log_std) + b_m
            x = self.activation(tf.einsum("...ni,...io->...no", x, w) + b)

        # Last layer has no activation function
        w_m, w_log_std, b_m, b_log_std = self.vars[-1]

        # Get noise
        z_w = self.noise_sampler(
            tf.concat([tf.shape(x)[:-2], tf.shape(w_log_std)], 0))
        z_b = self.noise_sampler(
            tf.concat([tf.shape(x)[:-2], [1],
                       tf.shape(b_log_std)], 0))

        w = z_w * tf.math.exp(w_log_std) + w_m
        b = z_b * tf.math.exp(b_log_std) + b_m
        return tf.einsum("...ni,...io->...no", x, w) + b
