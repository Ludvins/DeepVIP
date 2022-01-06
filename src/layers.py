import tensorflow as tf

from .utils import reparameterize

import tensorflow_probability as tfp
import numpy as np


class Layer(tf.keras.layers.Layer):
    def __init__(self, input_dim=None, dtype=None, seed=0):
        """
        A base class for VIP layers. Basic functionality for multisample
        conditional and input propagation.

        Parameters
        ----------
        input_dim : int
                    Input dimension
        dtype : data-type
                The dtype of the layer's computations and weights.
                Refer to tf.keras.layers.Layer for more information.
        seed : int
               integer to use as seed for randomness.
        """
        super().__init__(dtype=dtype)
        self.seed = seed
        self.build(input_dim)

    def train_mode(self):
        self.training = True
        self.generative_function.train_mode()

    def eval_mode(self):
        self.training = False
        self.generative_function.eval_mode()

    def conditional_ND(self, X, full_cov=False):
        """
        Computes the conditional probability \(Q^{\tar}(y \mid x, \theta)\).

        Parameters
        ----------
        X : tf.tensor of shape (N, D)
            Contains the input locations.

        full_cov : boolean
                   Wether to use full covariance matrix or not.
                   Determines the shape of the variance output.

        Returns
        -------
        mean : tf.tensor of shape (N, num_outputs)
               Contains the mean value of the distribution for each input

        var : tf.tensor of shape (N, num_outputs) or (N, N, num_outputs)
              Contains the variance value of the distribution for each input
        """
        raise NotImplementedError

    def KL(self):
        """
        The KL divergence from the variational distribution to the prior.
        """
        raise NotImplementedError

    def sample_from_conditional(self, X, z=None, full_cov=False):
        """
        Calculates self.conditional and also draws a sample.
        Adds input propagation if necessary.

        Parameters
        ----------
        X : tf.tensor of shape (N, D_in)
            Input locations.

        z : tf.tensor of shape (N, D)
            Contains a sample from a Gaussian distribution, ideally from a
            standardized Gaussian.

        full_cov : boolean
                   Wether to compute or not the full covariance matrix or just
                   the diagonal.

        Returns
        -------
        samples : tf.tensor of shape (S, N, self.num_outputs)
                  Samples from a Gaussian given by mean and var. See below.
        mean : tf.tensor of shape (S, N, self.num_outputs)
               Stacked tensor of mean values from conditional_ND applied to X.
        var : tf.tensor of shape (S, N, self.num_outputs or
                                  S, N, N, self.num_outputs)
              Stacked tensor of variance values from conditional_ND applied to
              X.
        """
        S, N, D = tf.shape(X)[0], tf.shape(X)[1], tf.shape(X)[2]
        X_flat = tf.reshape(X, [S * N, D])
        mean, var = self.conditional_ND(X_flat)

        mean = tf.reshape(mean, [S, N, self.output_dim])
        var = tf.reshape(var, [S, N, self.output_dim])

        # If no sample is given, generate it from a standardized Gaussian
        if z is None:
            z = tf.random.normal(
                shape=tf.shape(mean), seed=self.seed, dtype=self.dtype
            )
        # Apply re-parameterization trick to z
        samples = reparameterize(mean, var, z, full_cov=full_cov)

        return samples, mean, var


class VIPLayer(Layer):
    def __init__(
        self,
        generative_function,
        num_regression_coeffs,
        output_dim,
        input_dim,
        q_mu_initial_value,
        q_sqrt_initial_value,
        log_layer_noise=-5,
        seed=0,
        mean_function=None,
        add_prior_regularization=False,
        dtype=tf.float64,
    ):
        """
        A variational implicit process layer.

        The underlying model performs a Bayesian linear regression
        approximation
        \[
            f(x) = mean_function(x) + a^T \phi(x)
        \]
        with
        \[
            \phi(x) = \frac{1}{\sqrt(S)}(f_1(x) - m(x), \dots, f_S(x) - m(x))
        \]
        \[
            a \sim \mathcal{N}(0, I)
        \]
        Where S randomly sampled functions are used and m(x) denotes
        their empirical mean.

        The variational distribution over the regression coefficients is
        \[
            Q(a) = \mathcal{N}(a_mu, a_sqrt a_sqrt^T)
        \]

        The layer holds D_out independent VIPs with the same kernel
        and regression coefficients.


        Parameters
        ----------
        generative_function : GenerativeFunction
                              Generates function samples using the input
                              locations X and noise values.

        num_regression_coeffs : integer
                                Indicates the amount of linear regression
                                coefficients to use in the approximation.
                                Coincides with the number of samples of f.

        output_dim : int
                      The number of independent VIP in this layer.
                      More precisely, q_mu has shape (S, output_dim)

        input_dim : int
                    Dimensionality of the given features. Used to
                    pre-fix the shape of the different layers of the model.

        log_layer_noise : float or tf.tensor of shape (output_dim)
                          Contains the noise of each VIP contained in this
                          layer, i.e, epsilon ~ N(0, exp(log_layer_noise))

        mean_function : callable
                        Mean function added to the model. If no mean function
                        is specified, no value is added.

        trainable : boolean
                    Determines whether the regression parameters
                    q_mu and q_sqrt are trainable or not.

        dtype : data-type
                The dtype of the layer's computations and weights.
                Refer to tf.keras.layers.Layer for more information.
        """
        super().__init__(dtype=dtype, input_dim=input_dim, seed=seed)

        self.num_coeffs = num_regression_coeffs
        self.prior_regularization = add_prior_regularization
        # Regression Coefficients prior mean
        self.q_mu = tf.Variable(
            np.ones((self.num_coeffs, output_dim)) * q_mu_initial_value,
            name="q_mu",
        )

        # If no mean function is given, constant 0 is used
        self.mean_function = mean_function

        # Verticality of the layer
        self.output_dim = tf.constant(output_dim, dtype=tf.int32)

        # Initialize generative function
        self.generative_function = generative_function

        # Initialize the layer's noise
        if log_layer_noise is None:
            self.log_layer_noise = tf.cast(-13.0, dtype=self.dtype)
        else:
            self.log_layer_noise = tf.Variable(
                np.ones(self.output_dim) * log_layer_noise,
                name="layer_log_noise",
            )

        # Define Regression coefficients deviation using tiled triangular
        # identity matrix
        # Shape (num_coeffs, num_coeffs)
        q_var = np.eye(self.num_coeffs) * q_sqrt_initial_value
        # Replicate it output_dim times
        # Shape (output_dim, num_coeffs, num_coeffs)
        q_var = np.tile(q_var[None, :, :], [output_dim, 1, 1])
        # Create tensor with triangular representation.
        # Shape (output_dim, num_coeffs*(num_coeffs + 1)/2)
        q_sqrt_tri_prior = tfp.math.fill_triangular_inverse(q_var)
        self.q_sqrt_tri = tf.Variable(q_sqrt_tri_prior, name="q_sqrt_tri")

    @tf.function
    def conditional_ND(self, X, full_cov=False):
        """
        Computes Q*(y|x, a) using the linear regression approximation.
        Given that this distribution is Gaussian and Q(a) is also Gaussian
        the linear regression coefficients, a, can be marginalized, raising a
        Gaussian distribution as follows:

        Let

        phi(x) = 1/sqrt{S}(f_1(x) - m^*(x),...,f_S(\bm x) - m^*(x)),

        with f_1,..., f_S the sampled functions. Then if

        Q^*(y|x,a,\theta) = N(m^*(x) + 1/sqrt{S} phi(x)^T a, sigma^2)

        and

        Q(a) = N(q_mu, q_sqrt q_sqrt^T)

        the marginalized distribution is

        Q^*(y | x, \theta) = N(
            m^*(x) + 1/sqrt{S} phi(x)^T q_mu,
            sigma^2 + phi(x)^T q_sqrt q_sqrt^T phi(x)
        )

        Parameters:
        -----------
        X : tf.tensor of shape (N, D)
            Contains the input locations.

        full_cov : boolean
                   Wether to use full covariance matrix or not.
                   Determines the shape of the variance output.

        Returns:
        --------
        mean : tf.tensor of shape (N, output_dim)
               Contains the mean value of the distribution for each input

        var : tf.tensor of shape (N, output_dim) or (N, N, output_dim)
              Contains the variance value of the distribution for each input
        """

        # Let S = num_coeffs, D = output_dim and N = num_samples

        # Shape (S, M, N, D)
        f = self.generative_function(X, self.num_coeffs)

        # Compute mean value, shape (1, N, D)
        m = tf.reduce_mean(f, axis=0, keepdims=True)
        inv_sqrt = 1 / tf.math.sqrt(
            tf.cast(self.num_coeffs - 1, dtype=self.dtype)
        )
        # Compute regresion function, shape (S, N, D)
        phi = inv_sqrt * (f - m)

        # Compute mean value as m + q_mu^T phi per point and output dim
        # q_mu has shape (S, D)
        # phi has shape (S, ..., N, D)
        mean = tf.squeeze(m, axis=0) + tf.einsum("snd,sd->nd", phi, self.q_mu)
        # Shape (D, S, S)
        q_sqrt = tfp.math.fill_triangular(self.q_sqrt_tri)
        # Shape (D, S, S)
        Delta = tf.matmul(q_sqrt, q_sqrt, transpose_b=True)
        # Shape (S, S, D)
        Delta = tf.transpose(Delta, (1, 2, 0))

        # Compute variance matrix in two steps
        # Compute phi^T Delta = phi^T s_qrt q_sqrt^T
        K = tf.einsum("snd,skd->knd", phi, Delta)
        if full_cov:
            # var shape (num_points, num_points, output_dim)
            # Multiply by phi again distinguishing data_points
            K = tf.einsum("snd,smd->nmd", K, phi)
        else:
            # var shape (num_points, output_dim)
            # Multiply by phi again, using the same points twice
            K = tf.einsum("snd,snd->nd", K, phi)

        # Add layer noise to variance
        var = K + tf.math.exp(self.log_layer_noise)
        # Add mean function
        if self.mean_function is not None:
            mean = mean + self.mean_function(X)

        return mean, var

    @tf.function
    def KL(self):
        """
        Computes the KL divergence from the variational distribution of
        the linear regression coefficients to the prior.

        That is from a Gaussian N(q_mu, q_sqrt) to N(0, I).
        Uses formula for computing KL divergence between two
        multivariate normals, which in this case is:

        KL = 0.5 * ( tr(q_sqrt^T q_sqrt) +
                     q_mu^T q_mu - M - log |q_sqrt^T q_sqrt| )
        """

        D = tf.cast(self.output_dim, dtype=self.dtype)

        # Recover triangular matrix from array
        q_sqrt = tfp.math.fill_triangular(self.q_sqrt_tri)
        # Constant dimensionality term
        KL = -0.5 * D * self.num_coeffs
        # Log of determinant of covariance matrix.
        # Det(Sigma) = Det(q_sqrt q_sqrt^T) = Det(q_sqrt) Det(q_sqrt^T)
        #            = prod(diag_s_sqrt)^2
        KL -= tf.reduce_sum(
            tf.math.log(tf.math.abs(tf.linalg.diag_part(q_sqrt)))
        )

        # Trace term
        KL += 0.5 * tf.reduce_sum(tf.math.square(self.q_sqrt_tri))

        # Mean term
        KL += 0.5 * tf.reduce_sum(tf.math.square(self.q_mu))

        if self.prior_regularization:
            KL += self.generative_function.KL()

        return KL
