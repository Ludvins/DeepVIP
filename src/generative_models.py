import tensorflow as tf
import numpy as np


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

    def reset_seed(self):
        self.rng.reset_from_seed(self.seed)

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
        return self.rng.normal(size, dtype=tf.float64)


class UniformSampler(NoiseSampler):
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

    def reset_seed(self):
        self.rng.reset_from_seed(self.seed)

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
        return self.rng.uniform(size, dtype=tf.float64)


class GenerativeFunction(tf.Module):
    def __init__(
        self,
        input_dim,
        output_dim,
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
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.seed = seed

    def train_mode(self):
        self.training = True

    def eval_mode(self):
        self.training = False

    def __call__(self):
        """
        Generates the function samples.
        """
        raise NotImplementedError


class BayesianNN(GenerativeFunction):
    def __init__(
        self,
        structure,
        activation,
        output_dim,
        input_dim,
        seed,
        fix_random_noise,
        dropout,
        zero_mean_prior,
        dtype,
        w_mean_prior=0.0,
        w_std_prior=0.0,
        b_mean_prior=0.0,
        b_std_prior=0.0,
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
        self.initializer = tf.random_normal_initializer(
            mean=0, stddev=1.0, seed=seed
        )
        self.fix_random_noise = fix_random_noise
        self.structure = structure
        self.w_mean_prior = w_mean_prior
        self.w_std_prior = w_std_prior
        self.b_mean_prior = b_mean_prior
        self.b_std_prior = b_std_prior
        self.output_dim = output_dim
        self.seed = seed
        self.dtype = dtype

        self.dropout = dropout
        self.dropout_layer = tf.keras.layers.Dropout(
            dropout, seed=self.seed, dtype=self.dtype
        )

        self.noise_sampler = GaussianSampler(seed)

        self.activation = activation
        super().__init__(input_dim, output_dim, seed)
        dims = [input_dim] + self.structure + [self.output_dim]
        weights = []

        for _in, _out in zip(dims, dims[1:]):
            w_mean = tf.Variable(
                initial_value=self.initializer(shape=(_in, _out), dtype=dtype),
                name="w_mean_" + str(_in) + "-" + str(_out),
            )
            w_log_std = tf.Variable(
                initial_value=self.initializer(shape=(_in, _out), dtype=dtype),
                name="w_log_std_" + str(_in) + "-" + str(_out),
            )
            b_mean = tf.Variable(
                initial_value=self.initializer(shape=[_out], dtype=dtype),
                name="b_mean_" + str(_in) + "-" + str(_out),
            )
            b_log_std = tf.Variable(
                initial_value=self.initializer(shape=[_out], dtype=dtype),
                name="b_log_std_" + str(_in) + "-" + str(_out),
            )

            weights.append((w_mean, w_log_std, b_mean, b_log_std))

        # This has to be done this way or keras does not correctly add
        # the variables to the trainable_variables array
        self.vars = weights

    def __call__(self, inputs, num_samples):
        """
        Computes the output of the Bayesian neural network given the input
        values.
        """

        # inputs shape (N, D_in)
        x = tf.tile(tf.expand_dims(inputs, 0), [num_samples, 1, 1])
        if self.fix_random_noise:
            self.noise_sampler.reset_seed()

        for (w_m, w_log_std, b_m, b_log_std) in self.vars[:-1]:
            # Get noise

            # (S, M, D_out, D_in)
            z_w = self.noise_sampler(
                tf.concat([[num_samples], tf.shape(w_log_std)], 0)
            )
            # (S, M, 1, D_out)
            z_b = self.noise_sampler(
                tf.concat([[num_samples, 1], tf.shape(b_log_std)], 0)
            )

            # Compute Gaussian samples
            w = z_w * tf.math.exp(w_log_std) + w_m
            b = z_b * tf.math.exp(b_log_std) + b_m
            x = self.activation(tf.einsum("sni,sio->sno", x, w) + b)
            if self.dropout > 0.0 and self.training is True:
                x = self.dropout_layer(x, training=self.training)

        # Last layer has no activation function
        w_m, w_log_std, b_m, b_log_std = self.vars[-1]
        z_w = self.noise_sampler(
            tf.concat([[num_samples], tf.shape(w_log_std)], 0)
        )
        # (S, M, 1, D_out)
        z_b = self.noise_sampler(
            tf.concat([[num_samples, 1], tf.shape(b_log_std)], 0)
        )

        w = z_w * tf.math.exp(w_log_std) + w_m
        b = z_b * tf.math.exp(b_log_std) + b_m
        return tf.einsum("sni,sio->sno", x, w) + b


class BNN_GP(GenerativeFunction):
    def __init__(
        self,
        input_dim,
        output_dim,
        inner_layer_dim,
        dropout,
        seed,
        fix_random_noise,
        dtype,
    ):
        super().__init__(
            input_dim,
            output_dim,
            seed=seed,
        )

        self.fix_random_noise = fix_random_noise
        self.dtype = dtype
        self.inner_layer_dim = inner_layer_dim
        self.gaussian_sampler = GaussianSampler(seed)
        self.uniform_sampler = UniformSampler(seed=seed)
        self.dropout = dropout
        self.dropout_layer = tf.keras.layers.Dropout(
            dropout, seed=self.seed, dtype=self.dtype
        )

        self.initializer = tf.random_normal_initializer(
            mean=0, stddev=1.0, seed=seed
        )

        self.w_mean_1 = tf.Variable(
            initial_value=self.initializer(
                shape=(self.input_dim, self.inner_layer_dim), dtype=dtype
            ),
            name="w_mean_1",
        )
        self.w_log_std_1 = tf.Variable(
            initial_value=self.initializer(
                shape=(self.input_dim, self.inner_layer_dim), dtype=dtype
            ),
            name="w_log_std_1",
        )

        self.w_mean_2 = tf.Variable(
            initial_value=self.initializer(
                shape=(self.inner_layer_dim, self.output_dim), dtype=dtype
            ),
            name="w_mean_2",
        )
        self.w_log_std_2 = tf.Variable(
            initial_value=self.initializer(
                shape=(self.inner_layer_dim, self.output_dim), dtype=dtype
            ),
            name="w_log_std_2",
        )

        self.b_mean_2 = (
            tf.Variable(
                initial_value=self.initializer(
                    shape=[self.output_dim], dtype=dtype
                ),
                name="b_mean_2",
            ),
        )
        self.b_log_std_2 = tf.Variable(
            self.initializer(shape=[self.output_dim], dtype=dtype),
            name="b_log_std_2",
        )

        self.b_low = tf.Variable(
            initial_value=0.0, name="b_low", dtype=self.dtype
        )
        self.b_high = tf.Variable(
            initial_value=2 * np.pi, name="b_high", dtype=self.dtype
        )

    def __call__(self, inputs, num_samples):
        x = tf.tile(
            tf.expand_dims(inputs, 0),
            (num_samples, 1, 1),
        )
        if self.dropout > 0.0 and self.training:
            x = self.dropout_layer(x, training=self.training)

        if self.fix_random_noise:
            self.gaussian_sampler.reset_seed()
            self.uniform_sampler.reset_seed()

        z = self.gaussian_sampler(
            (num_samples, self.input_dim, self.inner_layer_dim)
        )
        b = (self.b_high - self.b_low) * self.uniform_sampler(
            (num_samples, 1, self.inner_layer_dim)
        ) + self.b_low

        w = self.w_mean_1 + z * tf.math.exp(self.w_log_std_1)
        x = tf.cast(
            tf.math.sqrt(2 / self.inner_layer_dim), dtype=self.dtype
        ) * tf.math.cos(x @ w + b)
        if self.dropout > 0.0 and self.training:
            x = self.dropout_layer(x, training=self.training)

        z_w = self.gaussian_sampler(
            (num_samples, self.inner_layer_dim, self.output_dim)
        )
        z_b = self.gaussian_sampler((num_samples, 1, self.output_dim))
        w = self.w_mean_2 + z_w * tf.math.exp(self.w_log_std_2)
        b = self.b_mean_2 + z_b * tf.math.exp(self.b_log_std_2)

        return x @ w + b
