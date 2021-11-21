import torch
from .utils import reparameterize
import numpy as np


class Layer(torch.nn.Module):
    def __init__(self, input_dim=None, dtype=None, device=None, seed=0):
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
        **kwargs : dict, optional
                   Extra arguments to `Layer`.
                   Refer to tf.keras.layers.Layer for more information.
        """
        super().__init__()
        self.seed = seed
        self.dtype = dtype
        self.device = device
        self.input_dim = input_dim

    def conditional_ND(self, X, full_cov=False):
        """
        Computes the conditional probability \(Q^{\star}(y \mid x, \theta)\).

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

    def conditional_SND(self, X, full_cov=False):
        """
        A multisample conditional, where X is shape (S,N,D_out), independent over samples S

        if full_cov is True
            mean is (S,N,D_out), var is (S,N,N,D_out)

        if full_cov is False
            mean and var are both (S,N,D_out)

        :param X:  The input locations (S,N,D_in)
        :param full_cov: Whether to calculate full covariance or just diagonal
        :return: mean (S,N,D_out), var (S,N,D_out or S,N,N,D_out)
        """

        S, N, D = X.shape

        if full_cov:
            raise NotImplementedError
        else:

            X_flat = X.reshape([S * N, D])
            mean, var, f = self.conditional_ND(X_flat)

        num_outputs = self.num_outputs

        mean = mean.reshape((S, N, num_outputs))
        f = f.reshape((S, N, -1, num_outputs))
        if full_cov:
            var = var.reshape((S, N, N, num_outputs))
        else:
            var = var.reshape((S, N, num_outputs))

        return mean, var, f

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

        mean, var, f = self.conditional_SND(X, full_cov=full_cov)

        if z is None:
            z = torch.randn(mean.shape).to(self.dtype).to(self.device)
        samples = reparameterize(mean, var, z, full_cov=full_cov)

        return samples, mean, var, f


class VIPLayer(Layer):
    def __init__(self,
                 generative_function,
                 num_regression_coeffs,
                 num_outputs,
                 input_dim,
                 log_layer_noise=-5,
                 mean_function=None,
                 trainable=True,
                 dtype=torch.float64,
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
                        Mean function added to the model. If no mean function
                        is specified, no value is added.

        trainable : boolean
                    Determines whether the regression parameters
                    q_mu and q_sqrt are trainable or not.

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
        self.q_mu = torch.nn.Parameter(
            torch.tensor(np.zeros((self.num_coeffs, num_outputs)),
                         dtype=self.dtype,
                         device=self.device))

        # If no mean function is given, constant 0 is used
        self.mean_function = mean_function

        # Verticality of the layer
        self.num_outputs = torch.tensor(num_outputs, dtype=torch.int32)

        # Initialize generative function
        self.generative_function = generative_function

        # Initialize the layer's noise
        self.log_layer_noise = torch.nn.Parameter(
            torch.tensor(np.ones(num_outputs) * log_layer_noise,
                         dtype=self.dtype,
                         device=self.device))

        # Define Regression coefficients deviation using tiled triangular
        # identity matrix
        # Shape (num_coeffs, num_coeffs)
        q_sqrt = np.eye(self.num_coeffs)
        # Replicate it num_outputs times
        # Shape (num_outputs, num_coeffs, num_coeffs)
        q_sqrt = np.tile(q_sqrt[None, :, :], [num_outputs, 1, 1])
        # Create tensor with triangular representation.
        # Shape (num_outputs, num_coeffs*(num_coeffs + 1)/2)
        li, lj = torch.tril_indices(self.num_coeffs, self.num_coeffs)
        triangular_q_sqrt = q_sqrt[:, li, lj]
        self.q_sqrt_tri = torch.nn.Parameter(
            torch.tensor(triangular_q_sqrt,
                         dtype=self.dtype,
                         device=self.device))

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

        # Let S = num_coeffs, D = num_outputs and N = num_samples

        # Shape (N, S, D)
        f = self.generative_function(X)

        # Compute mean value, shape (N, 1, D)
        m = torch.mean(f, dim=1, keepdims=True)

        inv_sqrt = 1 / torch.sqrt(
            torch.tensor(self.num_coeffs).type(self.dtype))
        # Compute regresion function, shape (N, S, D)
        phi = inv_sqrt * (f - m)

        # Compute mean value as m + q_mu^T phi per point and output dim
        # q_mu has shape (S, D)
        # phi has shape (N, S, D)
        mean = m.squeeze(axis=1) + torch.einsum("nsd,sd->nd", phi, self.q_mu)

        # Shape (D, S, S)
        q_sqrt = torch.zeros((self.num_outputs, self.num_coeffs,
                              self.num_coeffs)).to(self.dtype).to(self.device)
        li, lj = torch.tril_indices(self.num_coeffs, self.num_coeffs)
        q_sqrt[:, li, lj] = self.q_sqrt_tri

        # Shape (D, S, S)
        Delta = torch.matmul(q_sqrt, torch.transpose(q_sqrt, 1, 2))
        # Shape (S, S, D)
        Delta = torch.permute(Delta, (1, 2, 0))

        # Compute variance matrix in two steps
        # Compute phi^T Delta = phi^T s_qrt q_sqrt^T
        K = torch.einsum("nsd,skd->nkd", phi, Delta)

        if full_cov:
            # var shape (num_points, num_points, num_outputs)
            # Multiply by phi again distinguishing data_points
            K = torch.einsum("nsd,msd->nmd", K, phi)
        else:
            # var shape (num_points, num_outputs)
            # Multiply by phi again, using the same points twice
            K = torch.einsum("nsd,nsd->nd", K, phi)

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

        # Recover triangular matrix from array
        q_sqrt = torch.zeros((self.num_outputs, self.num_coeffs,
                              self.num_coeffs)).to(self.dtype).to(self.device)
        li, lj = torch.tril_indices(self.num_coeffs, self.num_coeffs)
        q_sqrt[:, li, lj] = self.q_sqrt_tri

        diag = torch.diagonal(q_sqrt, dim1=1, dim2=2)

        # Constant dimensionality term
        KL = -0.5 * self.num_outputs * self.num_coeffs

        # Log of determinant of covariance matrix.
        # Uses that sqrt(det(X)) = det(X^(1/2)) and
        # that the determinant of a upper triangular matrix (which q_sqrt is),
        # is the product of the diagonal entries (i.e. sum of their logarithm).
        KL -= 0.5 * torch.sum(2 * torch.log(diag))

        # Trace term.
        KL += 0.5 * torch.sum(torch.square(self.q_sqrt_tri))

        # Mean term
        KL += 0.5 * torch.sum(torch.square(self.q_mu))

        return KL
