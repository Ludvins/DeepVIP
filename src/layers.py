import numpy as np
import torch

from .utils import reparameterize


class Layer(torch.nn.Module):
    def __init__(self, input_dim=None, dtype=None, device=None, seed=0):
        """
        A base class for VIP layers. Basic functionality for multisample
        conditional.

        Parameters
        ----------
        input_dim : int
                    Input dimension
        dtype : data-type
                The dtype of the layer's computations and weights.
        device : torch.device
                 The device in which the computations are made.
        seed : int
               integer to use as seed for randomness.
        """
        super().__init__()
        self.seed = seed
        self.dtype = dtype
        self.device = device
        self.input_dim = input_dim
        self.freeze = False

        self.generator = torch.Generator(device)
        self.generator.manual_seed(seed)

    def conditional_ND(self):
        """
        Computes the conditional probability Q*(y | x, theta).
        """
        raise NotImplementedError

    def KL(self):
        """
        The KL divergence from the variational distribution to the prior.
        """
        raise NotImplementedError

    def sample_from_conditional(self, X, full_cov=False):
        """
        Calculates self.conditional and also draws a sample.
        Adds input propagation if necessary.

        Parameters
        ----------
        X : torch tensor of shape (..., N, D_in)
            Input locations. Additional dimensions at the beggining are
            propagated over each computation.
        full_cov : boolean
                   Wether to compute or not the full covariance matrix or just
                   the diagonal.
        Returns
        -------
        samples : torch tensor of shape (..., N, self.output_dim)
                  Samples from a Gaussian given by mean and var.
        mean : torch tensor of shape (..., N, self.output_dim)
               Mean values from conditional_ND applied to X.
        var : torch tensor of shape (..., N , self.output_dim)
              Variance values from conditional_ND applied to X.
        prior_samples : torch tensor of shape (self.num_coeffs, ...,
                        N, self.output_dim)
                        Prior samples from conditional_ND applied to X.
        """

        S, N, D = X.shape
        X_flat = torch.reshape(X, [S * N, D])
        mean, var, prior_samples = self.conditional_ND(
            X_flat, full_cov=full_cov
        )
        mean = torch.reshape(mean, [S, N, mean.shape[-1]])
        var = torch.reshape(var, [S, N, var.shape[-1]])
        prior_samples = torch.reshape(
            prior_samples,
            [prior_samples.shape[0], S, N, prior_samples.shape[-1]],
        )

        z = torch.randn(
            mean.shape,
            generator=self.generator,
            dtype=self.dtype,
            device=self.device,
        )
        samples = reparameterize(mean, var, z, full_cov=full_cov)
        return samples, mean, var, prior_samples


class VIPLayer(Layer):
    def __init__(
        self,
        generative_function,
        num_regression_coeffs,
        input_dim,
        output_dim,
        add_prior_regularization=False,
        log_layer_noise=None,
        q_sqrt_initial_value=1,
        q_mu_initial_value=0,
        mean_function=None,
        seed=0,
        dtype=torch.float64,
        device=None,
    ):
        """
        A variational implicit process layer.

        The underlying model performs a Bayesian linear regression
        approximation

        f(x) = mean_function(x) + a^T \phi(x)

        with

        phi(x) = 1/S (f_1(x) - m(x), ..., f_S(x) - m(x)),  a ~ N(0, I)

        Where S randomly sampled functions are used and m(x) denotes
        their empirical mean.

        The variational distribution over the regression coefficients is

            Q(a) = N(a_mu, a_sqrt a_sqrt^T)

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
        input_dim : int
                    Dimensionality of the given features. Used to
                    pre-fix the shape of the different layers of the model.
        output_dim : int
                      The number of independent VIP in this layer.
                      More precisely, q_mu has shape (S, output_dim)
        log_layer_noise : float or tf.tensor of shape (output_dim)
                          Contains the noise of each VIP contained in this
                          layer, i.e, epsilon ~ N(0, exp(log_layer_noise))
        mean_function : callable
                        Mean function added to the model. If no mean function
                        is specified, no value is added.
        dtype : data-type
                The dtype of the layer's computations and weights.
                Refer to tf.keras.layers.Layer for more information.
        seed : int
               integer to use as seed for randomness.
        """
        super().__init__(
            dtype=dtype, input_dim=input_dim, seed=seed, device=device
        )
        self.add_prior_regularization = add_prior_regularization

        self.num_coeffs = num_regression_coeffs

        # Regression Coefficients prior mean
        self.q_mu = torch.tensor(
            np.ones((self.num_coeffs, output_dim)) * q_mu_initial_value,
            dtype=self.dtype,
            device=self.device,
        )
        self.q_mu = torch.nn.Parameter(self.q_mu)

        # If no mean function is given, constant 0 is used
        self.mean_function = mean_function

        # Verticality of the layer
        self.output_dim = torch.tensor(
            output_dim, dtype=torch.int32, device=device
        )

        # Initialize generative function
        self.generative_function = generative_function

        # Initialize the layer's noise
        if log_layer_noise is not None:
            self.log_layer_noise = torch.nn.Parameter(
                torch.tensor(
                    np.ones(output_dim) * log_layer_noise,
                    dtype=self.dtype,
                    device=self.device,
                )
            )
        else:
            self.log_layer_noise = torch.tensor(0)

        # Define Regression coefficients deviation using tiled triangular
        # identity matrix
        # Shape (num_coeffs, num_coeffs)
        q_sqrt = np.eye(self.num_coeffs) * q_sqrt_initial_value
        # Replicate it output_dim times
        # Shape (num_coeffs, num_coeffs, output_dim)
        q_sqrt = np.tile(q_sqrt[:, :, None], [1, 1, output_dim])
        # Create tensor with triangular representation.
        # Shape (output_dim, num_coeffs*(num_coeffs + 1)/2)
        li, lj = torch.tril_indices(self.num_coeffs, self.num_coeffs)
        triangular_q_sqrt = q_sqrt[li, lj]
        self.q_sqrt_tri = torch.tensor(
            triangular_q_sqrt,
            dtype=self.dtype,
            device=self.device,
        )
        self.q_sqrt_tri = torch.nn.Parameter(self.q_sqrt_tri)

    def freeze_posterior(self):
        """Sets the model parameters as non-trainable."""
        self.q_mu.requires_grad = False
        self.q_sqrt_tri.requires_grad = False
        self.log_layer_noise.requires_grad = False

    def freeze_prior(self):
        """Sets the prior parameters of this layer as non trainable."""
        self.generative_function.freeze_parameters()

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

        Q(a) = N(q_mu, q_sqrt q_sqrt^T)

        the marginalized distribution is

        Q^*(y | x, \theta) = N(
            m^*(x) + 1/sqrt{S} phi(x)^T q_mu,
            sigma^2 + phi(x)^T q_sqrt q_sqrt^T phi(x)
        )

        Parameters:
        -----------
        X : torch tensor of shape (..., N, D)
            Contains the input locations.
        full_cov : boolean
                   Wether to use full covariance matrix or not.
                   Determines the shape of the variance output.
        Returns:
        --------
        mean : torch tensor of shape (..., N, self.output_dim)
               Mean values from conditional_ND applied to X.
        var : torch tensor of shape (..., N , self.output_dim)
              Variance values from conditional_ND applied to X.
        prior_samples : torch tensor of shape (self.num_coeffs, ...,
                        N, self.output_dim)
                        Prior samples from conditional_ND applied to X.
        """

        # Let S = num_coeffs, D = output_dim and N = num_samples
        # Shape (S, ... , N, D)
        f = self.generative_function(X, self.num_coeffs)
        # Compute mean value, shape (1, ... , N, D)
        m = torch.mean(f, dim=0, keepdims=True)

        # Compute regresion function, shape (S, ... , N, D)
        phi = (f - m) / torch.sqrt(
            torch.tensor(self.num_coeffs).type(self.dtype)
        )
        # Compute mean value as m + q_mu^T phi per point and output dim
        # q_mu has shape (S, D)
        # phi has shape (S, ... , N, D)
        mean = m.squeeze(axis=0) + torch.einsum("snd,sd->nd", phi, self.q_mu)

        # Shape (S, S, D)
        q_sqrt = (
            torch.zeros((self.num_coeffs, self.num_coeffs, self.output_dim))
            .to(self.dtype)
            .to(self.device)
        )
        li, lj = torch.tril_indices(self.num_coeffs, self.num_coeffs)
        q_sqrt[li, lj] = self.q_sqrt_tri
        # Shape (S, S, D)
        Delta = torch.einsum("ijd, kjd -> ikd", q_sqrt, q_sqrt)

        # Compute variance matrix in two steps
        # Compute phi^T Delta = phi^T s_qrt q_sqrt^T
        K = torch.einsum("snd,skd->knd", phi, Delta)

        if full_cov:
            # var shape (num_points, num_points, output_dim)
            # Multiply by phi again distinguishing data_points
            K = torch.einsum("snd,smd->nmd", K, phi)
        else:
            # var shape (num_points, output_dim)
            # Multiply by phi again, using the same points twice
            K = torch.einsum("snd,snd->nd", K, phi)

        # Add layer noise to variance
        var = K + torch.exp(self.log_layer_noise)

        # Add mean function
        if self.mean_function is not None:
            mean = mean + self.mean_function(X)

        return mean, var, f

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

        # self.q_sqrt_tri stores the triangular matrix using indexes
        #  (0,0), (1,0), (1,1), (2,0), (2,1), (2,2), (3,0)....
        #  knowing this, the diagonal is stored at positions 0, 2, 5, 9, 13...
        #  which can be created using np.cumsum
        diag_indexes = np.cumsum(np.arange(1, self.num_coeffs + 1)) - 1
        diag = self.q_sqrt_tri[diag_indexes]
        # Constant dimensionality term
        KL = -0.5 * self.output_dim * self.num_coeffs

        # Log of determinant of covariance matrix.
        # Det(Sigma) = Det(q_sqrt q_sqrt^T) = Det(q_sqrt) Det(q_sqrt^T)
        #            = prod(diag_s_sqrt)^2
        KL -= torch.sum(torch.log(torch.abs(diag)))

        # Trace term
        KL += 0.5 * torch.sum(torch.square(self.q_sqrt_tri))

        # Mean term
        KL += 0.5 * torch.sum(torch.square(self.q_mu))

        if self.add_prior_regularization:
            KL += self.generative_function.KL()

        return KL
