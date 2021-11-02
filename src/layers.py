import tensorflow as tf

from .utils import reparameterize

import tensorflow_probability as tfp
import numpy as np


class Layer(tf.keras.layers.Layer):
    def __init__(self, input_dim=None, dtype=None, seed=0, **kwargs):
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


        **kwargs : dict, optional
                   Extra arguments to `Layer`.
                   Refer to tf.keras.layers.Layer for more information.
        """
        super().__init__(dtype=dtype, **kwargs)
        self.seed = seed
        self.build(input_dim)

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

    @tf.function
    def conditional_SND(self, X, full_cov=False):
        """
        A multisample conditional.

        Parameters
        ----------
        X : tf.tensor of shape (S, N, D_in)
            Input locations.
            S independent draws of N samples with dimension D_in
        full_cov : boolean
                   Wether to compute or not the full covariance matrix or just
                   the diagonal.

        Returns
        -------
        mean : tf.tensor of shape (S, N, self.num_outputs)
               Stacked tensor of mean values from conditional_ND applied to
               each sample.
        var : tf.tensor of shape (S, N, self.num_outputs or
                                  S, N, N, self.num_outputs)
              Stacked tensor of variance values from conditional_ND applied to
              each sample.
        """

        if full_cov:

            def f(a):
                return self.conditional_ND(a, full_cov=True)

            mean, var = tf.map_fn(f, X, dtype=(tf.float64, tf.float64))
            return tf.stack(mean), tf.stack(var)
        else:
            S = tf.shape(X)[0]
            N = tf.shape(X)[1]
            D = tf.shape(X)[2]

            X_flat = tf.reshape(X, [S * N, D])
            mean, var = self.conditional_ND(X_flat, full_cov=False)

            return [
                tf.reshape(m, [S, N, self.num_outputs]) for m in [mean, var]
            ]

    @tf.function
    def sample_from_conditional(self, X, z=None, full_cov=False):
        """
        Calculates self.conditional and also draws a sample.
        Adds input propagation if necessary.

        Parameters
        ----------
        X : tf.tensor of shape (S, N, D_in)
            Input locations.
            S independent draws of N samples with dimension D_in

        z : tf.tensor of shape (S, N, D)
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
        mean, var = self.conditional_ND(X, full_cov=full_cov)

        # If no sample is given, generate it from a standardized Gaussian
        if z is None:
            z = tf.random.normal(shape=tf.shape(mean),
                                 seed=self.seed,
                                 dtype=self.dtype)
        # Apply re-parameterization trick to z
        samples = reparameterize(mean, var, z, full_cov=full_cov)

        return samples, mean, var


class VIPLayer(Layer):
    def __init__(self,
                 noise_sampler,
                 generative_function,
                 num_regression_coeffs,
                 num_outputs,
                 input_dim,
                 log_layer_noise=-5,
                 mean_function=None,
                 dtype=tf.float64,
                 **kwargs):
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
        noise_sampler : NoiseSampler
                        Generates samples from a noise distribution.

        generative_function : GenerativeFunction
                              Generates function samples using the input
                              locations X and noise values.

        num_regression_coeffs : integer
                                Indicates the amount of linear regression
                                coefficients to use in the approximation.
                                Coincides with the number of samples of f.

        num_outputs : int
                      The number of independent VIP in this layer.
                      More precisely, q_mu has shape (S, num_outputs)

        input_dim : int
                    Dimensionality of the given features. Used to
                    pre-fix the shape of the different layers of the model.

        log_layer_noise : float or tf.tensor of shape (num_outputs)
                          Contains the noise of each VIP contained in this
                          layer, i.e, epsilon ~ N(0, exp(log_layer_noise))

        mean_function : callable
                        Mean function added to the model.

        dtype : data-type
                The dtype of the layer's computations and weights.
                Refer to tf.keras.layers.Layer for more information.

        **kwargs : dict, optional
                   Extra arguments to `Layer`.
                   Refer to tf.keras.layers.Layer for more information.
        """
        super().__init__(dtype=dtype, input_dim=input_dim, **kwargs)

        self.num_coeffs = num_regression_coeffs

        # Regression Coefficients prior mean
        q_mu = np.zeros((self.num_coeffs, num_outputs))
        self.q_mu = tf.Variable(q_mu, name="q_mu")

        # If no mean function is given, constant 0 is used
        self.mean_function = mean_function

        # Verticality of the layer
        self.num_outputs = tf.constant(num_outputs, dtype=tf.int32)

        # Initialize generative function
        self.generative_function = generative_function(
            noise_sampler=noise_sampler,
            num_samples=num_regression_coeffs,
            num_outputs=num_outputs,
            input_dim=input_dim,
            seed=self.seed,
        )

        # Initialize the layer's noise
        self.log_layer_noise = tf.Variable(
            initial_value=np.ones(num_outputs) * log_layer_noise,
            dtype="float64",
            trainable=True,
            name="layer_log_noise",
        )

        # Define Regression coefficients deviation using tiled triangular
        # identity matrix
        # Shape (num_coeffs, num_coeffs)
        identity = np.eye(self.num_coeffs)
        # Replicate it num_outputs times
        # Shape (num_outputs, num_coeffs, num_coeffs)
        I_tiled = np.tile(identity[None, :, :], [num_outputs, 1, 1])
        # Create tensor with triangular representation.
        # Shape (num_outputs, num_coeffs*(num_coeffs + 1)/2)
        triangular_I_tiled = tfp.math.fill_triangular_inverse(I_tiled)
        self.q_sqrt_tri = tf.Variable(triangular_I_tiled, name="q_sqrt_tri")

    @tf.function
    def conditional_ND(self, X, full_cov=False):
        """
        Computes Q*(y|x, a) using the linear regression approximation.
        Given that this distribution is Gaussian and Q(a) is also Gaussian
        the linear regression coefficients, a, can be marginalized, raising a
        Gaussian distribution as follows:

        Let

        phi(x) =   1/sqrt{S}(f_1(x) - m^*(x),...,f_S(\bm x) - m^*(x)),

        with f_1,..., f_S the sampled functions. Then if

        Q^*(y|x,a,\theta) = N(m^*(x) + 1/sqrt{S} phi(x)^T a, sigma^2)

        and

        Q(a) = N(q_mu, q_sigma q_sigma^T)

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
        mean : tf.tensor of shape (N, num_outputs)
               Contains the mean value of the distribution for each input

        var : tf.tensor of shape (N, num_outputs) or (N, N, num_outputs)
              Contains the variance value of the distribution for each input
        """

        # Shape (num_coeffs, N, D)
        f = self.generative_function(X)
        # Compute mean value, shape (N, D)
        m = tf.reduce_mean(f, axis=0)

        # Compute regresion function, shape (num_coeffs, N, D)
        inv_sqrt = 1 / tf.math.sqrt(tf.cast(self.num_coeffs, dtype="float64"))
        phi = inv_sqrt * (f - m)

        # Compute mean value as m + q_mu^T phi per point and output dim
        # q_mu has shape (num_coeffs, N, num_outputs)
        # phi has shape (num_coeffs, num_outputs)
        mean = m + tf.einsum("snd,sd->nd", phi, self.q_mu)

        # Shape (num_outputs, num_coeffs, num_coeffs)
        q_sqrt = tfp.math.fill_triangular(self.q_sqrt_tri)
        # Shape (num_outputs, num_coeffs, num_coeffs)
        Delta = tf.matmul(q_sqrt, q_sqrt, transpose_b=True)

        # Compute variance matrix in two steps
        # Compute phi^T Delta = phi^T s_qrt q_sqrt^T
        K = tf.einsum("snd,dsk->knd", phi, Delta)

        if full_cov:
            # var shape (num_points, num_points, num_outputs)
            # Multiply by phi again distinguishing data_points
            K = tf.einsum("snd,smd->nmd", K, phi)
        else:
            # var shape (num_points, num_outputs)
            # Multiply by phi again, using the same points twice
            K = tf.einsum("snd,snd->nd", K, phi)

        # Add layer noise to variance
        var = K + tf.math.exp(self.log_layer_noise)

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

        D = tf.cast(self.num_outputs, dtype=self.dtype)

        # Recover triangular matrix from array
        q_sqrt = tfp.math.fill_triangular(self.q_sqrt_tri)
        diag = tf.linalg.diag_part(q_sqrt)

        # Constant dimensionality term
        KL = -0.5 * D * self.num_coeffs

        # Log of determinant of covariance matrix.
        # Uses that sqrt(det(X)) = det(X^(1/2)) and
        # that the determinant of a upper triangular matrix (which q_sqrt is),
        # is the product of the diagonal entries (i.e. sum of their logarithm).
        KL -= 0.5 * tf.reduce_sum(2 * tf.math.log(diag))

        # Trace term.
        KL += 0.5 * tf.reduce_sum(tf.square(self.q_sqrt_tri))

        # Mean term
        KL += 0.5 * tf.reduce_sum(tf.square(self.q_mu))

        return KL

    def print_variables(self):
        print("Regression coefficients")
        print("Mean: ")
        print(self.q_mu.value().numpy())
        q_sqrt = tfp.math.fill_triangular(self.q_sqrt_tri)
        print("Covariance: ")
        print(q_sqrt)
        print("Generative model variables:")
        print(self.generative_function.trainable_variables)
