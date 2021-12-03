import torch
import numpy as np


class NoiseSampler:
    def __init__(self, seed, dtype=torch.float64):
        """
        Generates noise samples.

        Parameters:
        -----------
        seed : int
               Integer value used to generate reproducible results.
        dtype : data-type
                The dtype of the layer's computations and weights.
        """
        self.seed = seed
        self.dtype = dtype

    def call(self):
        """
        Returns sampled noise values.
        """
        raise NotImplementedError


class GaussianSampler(NoiseSampler):
    def __init__(self, seed=2147483647, dtype=torch.float64):
        """
        Generates noise samples from a Standar Gaussian distribution N(0, 1).

        Parameters:
        -----------
        seed : int
               Integer value used to generate reproducible results.
        dtype : data-type
                The dtype of the layer's computations and weights.
        """
        super().__init__(seed, dtype)
        self.generator = torch.Generator()
        self.generator.manual_seed(seed)

    def set_seed(self):
        self.generator.manual_seed(self.seed)

    def __call__(self, size):
        """
        Returns sampled noise values os the given size or shape.

        Parameters:
        -----------
        size : int or np.darray
               Indicates the desired shape/size of the sample to generate.

        Returns:
        --------
        samples : torch tensor of shape (size)
                  A sample from a Gaussian distribution N(0, I).

        """
        return torch.randn(
            size=size,
            generator=self.generator,
            dtype=self.dtype,
        )


class GenerativeFunction(torch.nn.Module):
    def __init__(
        self, input_dim, output_dim, device=None, seed=0, dtype=torch.float64
    ):
        """
        Generates samples from a stochastic function using sampled
        noise values and input values.

        Parameters:
        -----------
        input_dim : int or array
                    Dimensionality of the input values `x`.
        output_dim : int
                     Dimensionality of the function output.
        dtype : data-type
                The dtype of the layer's computations and weights.
        """
        super(GenerativeFunction, self).__init__()
        self.output_dim = output_dim
        self.input_dim = input_dim
        self.device = device
        self.seed = seed
        self.dtype = dtype

    def freeze_parameters(self):
        for param in self.parameters():
            param.requires_grad = False

    def forward(self):
        """
        Generates the function samples.
        """
        raise NotImplementedError


class BayesLinear(GenerativeFunction):
    def __init__(
        self,
        input_dim,
        output_dim,
        w_mean_prior,
        w_log_std_prior,
        b_mean_prior,
        b_log_std_prior,
        device=None,
        seed=0,
        dtype=torch.float64,
    ):
        """
        Generates samples from a stochastic Bayesian Linear function
        f(x) = w^T x + b,   where w and b follow a Gaussian distribution,
        parameterized by their mean and log standard deviation.

        Parameters:
        -----------
        input_dim : int or array
                    Dimensionality of the input values `x`.
        num_outputs : int
                      Dimensionality of the function output.
        w_mean_prior : torch tensor
                       Prior value w's mean.
        w_log_std_prior : torch tensor
                          Logarithm of the prior standard deviation of w.
        b_mean_prior : torch tensor
                       Prior value b's mean.
        b_log_std_prior : torch tensor
                          Logarithm of the prior standard deviation of b.
        dtype : data-type
                The dtype of the layer's computations and weights.
        """
        super(BayesLinear, self).__init__(
            input_dim, output_dim, device=device, seed=seed, dtype=dtype
        )

        # Instantiate Standard Gaussian sampler
        self.gaussian_sampler = GaussianSampler(seed)

        # Store prior values
        self.w_mean_prior = torch.clone(w_mean_prior)
        self.w_log_std_prior = torch.clone(w_log_std_prior)
        self.b_mean_prior = torch.clone(b_mean_prior)
        self.b_log_std_prior = torch.clone(b_log_std_prior)

        # Create trainable parameters
        self.weight_mu = torch.nn.Parameter(w_mean_prior)
        self.weight_log_sigma = torch.nn.Parameter(w_log_std_prior)
        self.bias_mu = torch.nn.Parameter(b_mean_prior)
        self.bias_log_sigma = torch.nn.Parameter(b_log_std_prior)

    def forward(self, inputs):

        # Check the given input is valid
        if inputs.shape[-1] != self.input_dim:
            raise RuntimeError(
                "Input shape does not match stored data dimension"
            )

        # Given an input of shape (A1, A2, ..., AM, N, D), random values are
        # generated over every dimension but the batch one (N), that is,
        # z_w_shape is (A1, ..., AM, D, D_out) and z_b_shape is
        # (A1, ..., AM, 1, D_out)
        z_w_shape = (*inputs.shape[:-2], self.input_dim, self.output_dim)
        z_b_shape = (*inputs.shape[:-2], 1, self.output_dim)

        # self.gaussian_sampler.set_seed()

        # Generate Gaussian values
        z_w = self.gaussian_sampler(z_w_shape)
        z_b = self.gaussian_sampler(z_b_shape)

        # Perform reparameterization trick
        w = self.weight_mu + z_w * torch.exp(self.weight_log_sigma)
        b = self.bias_mu + z_b * torch.exp(self.bias_log_sigma)

        # Apply linear transformation, dotwise over the first dimension
        #  and matrix multiplication over the last one.
        return torch.einsum("...nd, ...do -> ...no", inputs, w) + b

    def KL(self):
        """
        Computes the KL divergence of w and b from their prior value.

        Returns
        -------
        KL : int
             The addition of the 2 KL terms computed
        """
        return torch.tensor(0.0)
        # Compute w's flattened mean and covariance diagonal matrix
        w_m = torch.flatten(self.weight_mu)
        w_Sigma = torch.flatten(torch.square(torch.exp(self.weight_log_sigma)))
        w_m_prior = torch.flatten(self.w_mean_prior)
        w_Sigma_prior = torch.flatten(
            torch.square(torch.exp(self.w_log_std_prior))
        )

        # Compute b's flattened mean and covariance diagonal matrix
        b_m = torch.flatten(self.bias_mu)
        b_Sigma = torch.flatten(torch.square(torch.exp(self.bias_log_sigma)))
        b_m_prior = torch.flatten(self.b_mean_prior)
        b_Sigma_prior = torch.flatten(
            torch.square(torch.exp(self.b_log_std_prior))
        )

        # Compute the KL divergence of w
        KL = -w_m.size(dim=0)
        KL += torch.sum(w_Sigma / w_Sigma_prior)
        KL += torch.sum((w_m_prior - w_m) ** 2 / w_Sigma_prior)
        KL += 2 * torch.sum(self.w_log_std_prior) - 2 * torch.sum(
            self.weight_log_sigma
        )

        # Compute the KL divergence of b
        KL -= b_m.size(dim=0)
        KL += torch.sum(b_Sigma / b_Sigma_prior)
        KL += torch.sum((b_m_prior - b_m) ** 2 / b_Sigma_prior)
        KL += 2 * torch.sum(self.b_log_std_prior) - 2 * torch.sum(
            self.bias_log_sigma
        )

        return KL / 2


class BayesianNN(GenerativeFunction):
    def __init__(
        self,
        structure,
        activation,
        input_dim=1,
        output_dim=1,
        seed=0,
        device=None,
        dtype=torch.float64,
    ):
        """
        Defines a Bayesian Neural Network.

        Parameters:
        -----------
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
        """
        super().__init__(
            input_dim, output_dim, device=device, seed=seed, dtype=dtype
        )

        self.structure = structure
        self.activation = activation
        self.generator = torch.Generator()
        self.generator.manual_seed(self.seed)

        # Create an array symbolizing the dimensionality of the data at
        # each inner layer.
        dims = [input_dim] + self.structure + [self.output_dim]
        layers = []

        # Loop over the input and output dimension of each sub-layer.
        for _in, _out in zip(dims, dims[1:]):
            layers.append(
                BayesLinear(
                    _in,
                    _out,
                    # Sampler the prior values from a Gaussian distribution
                    w_mean_prior=torch.normal(
                        mean=0.0,
                        std=0.01,
                        size=(_in, _out),
                        generator=self.generator,
                    ),
                    w_log_std_prior=torch.log(
                        torch.abs(
                            torch.normal(
                                mean=0.01,
                                std=0.5,
                                size=(_in, _out),
                                generator=self.generator,
                            )
                        )
                    ),
                    b_mean_prior=torch.normal(
                        mean=0.01,
                        std=0.1,
                        size=[_out],
                        generator=self.generator,
                    ),
                    b_log_std_prior=torch.log(
                        torch.abs(
                            torch.normal(
                                mean=0.0,
                                std=1.1,
                                size=[_out],
                                generator=self.generator,
                            )
                        )
                    ),
                    device=device,
                    dtype=dtype,
                )
            )
        self.layers = torch.nn.ModuleList(layers)

    def forward(self, inputs):
        """Forward pass over each inner layer, activation is applied on every
        but the last level.
        """
        x = inputs
        for layer in self.layers[:-1]:
            x = self.activation(layer(x))

        return self.layers[-1](x)

    def KL(self):
        """Computes the Kl divergence of the model as the addition of the
        KL divergences of its sub-models.
        """
        return torch.stack([layer.KL() for layer in self.layers]).sum()


def RBF(X1, X2, l=1):
    """Computes the RBF kernel of the given input torch tensors.

    Arguments
    ---------

    X1 : torch tensor of shape (..., N, D)
         First input tensor
    X2 : torch tensor of shape (..., N, D)
         Second input tensor

    Returns
    -------
    K : torch tensor of shape (..., N, N)
        xddContains the application of the rbf kernel.
    """
    # Shape (..., N, 1, D)
    X = X1.unsqueeze(-2)
    # Shape (..., 1, N, D)
    Y = X2.unsqueeze(-3)
    # X - Y has shape (..., N, N, D) due to broadcasting
    K = ((X - Y) ** 2).sum(-1)

    return torch.exp(-K / l ** 2)


class GP(GenerativeFunction):
    def __init__(
        self,
        X,
        y,
        kernel=RBF,
        noise=1e-6,
        device=None,
        seed=0,
        dtype=torch.float64,
    ):
        """Encapsulates an exact Gaussian process. The Cholesky decomposition
        of the kernel is used in order to speed up computations.

        Arguments
        ---------
        X : torch tensor of shape (N, D)
            Contains the learning inputs.
        y : torch tensor of shape (N, D_out)
            Contains the learning labels.
        kernel : callable
                 The desired kernel function to use, must accept batched
                 inputs (..., N, D) and compute the kernel matrix with shape
                 (..., N, N).
                 Defaults to RBF.
        noise : float
                Considered noise value in the Gaussian Process.
                Defaults to 1e-6.
        device : torch.device
                 The device in which the computations are made.
        dtype : data-type
                The dtype of the layer's computations and weights.
        """

        N, input_dim = X.shape
        output_dim = y.shape[-1]
        super(GP, self).__init__(
            input_dim, output_dim, device=device, seed=seed, dtype=dtype
        )

        # Instantiate Standard Gaussian sampler
        self.gaussian_sampler = GaussianSampler(seed)

        # Instantiate the kernel
        self.kernel = kernel

    def forward(self, inputs):
        """Creates a sample from the posterior distribution of the
        Gaussian process.

        Arguments
        ---------
        inputs : torch tensor of shape (..., M, D)
                 Batched input values

        Returns
        -------
        sample : torch tensor of shape (...., M, D_out)
                 Generated sample
        """

        # Compute the posterior mean
        #  Shape (..., M, D_out)
        mu = 0

        # Compute K(X*, X*)
        cov = self.kernel(inputs, inputs)

        # Create sample from posterior distribution using the cholesky
        #  decomposition of the covariance matrix
        #  shape (20, 3, N, N)
        L = torch.linalg.cholesky(cov + 1e-5 * torch.eye(cov.shape[-1]))
        # (20, 3, 1)

        z = self.gaussian_sampler(L.shape[:-1])
        z = z.unsqueeze(-1)

        # Shape (20, 3, N, 1)
        return mu + L @ z

    def KL(self):
        return 0.0


class GP_Inducing(GenerativeFunction):
    def __init__(
        self,
        input_dim,
        output_dim,
        kernel=RBF,
        noise=1e-6,
        inducing=None,
        num_inducing=10,
        device=None,
        seed=0,
        dtype=torch.float64,
    ):
        """Gaussian process with inducing points. This Model holds the kernel,
        variational parameters and inducing points.

        The underlying model at inputs X is
        f = Lv, where v \sim N(0, I) and LL^T = kern.K(X)

        The variational distribution over the inducing points is
        q(v) = N(q_mu, q_sqrt q_sqrt^T)

        The layer holds D_out independent GPs with the same kernel
        and inducing points.

        Arguments
        ---------
        input_dim : int or array
                    Dimensionality of the input values.
        output_dim : int
                     Dimensionality of the function output.
        kernel : callable
                 The desired kernel function to use, must accept batched
                 inputs (..., N, D) and compute the kernel matrix with shape
                 (..., N, N).
                 Defaults to RBF.
        noise : float
                Considered noise value in the Gaussian Process.
                Defaults to 1e-6.
        inducing : array-like of shape (M, D)
                   Contains the desired initialization of the inducing points.
                   If this parameter is given, num_inducing is not used.
        num_inducing : int
                       Number of inducing points to consider if not given.
        device : torch.device
                 The device in which the computations are made.
        dtype : data-type
                The dtype of the layer's computations and weights.
        """

        super(GP_Inducing, self).__init__(
            input_dim, output_dim, device=device, seed=seed, dtype=dtype
        )

        # If no inducing points are provided, initialice these to 0
        if inducing is None:
            inducing = np.zeros((num_inducing, output_dim))
        else:
            # If they are, check the dimensions are correct
            if output_dim != inducing.shape[1]:
                raise Exception(
                    "Labels dimension does not coincide"
                    " with inducing points dimension"
                )

        # Create torch tensor and Parameter for the inducing points
        inducing = torch.tensor(
            inducing,
            dtype=self.dtype,
            device=self.device,
        )
        self.inducing = torch.nn.Parameter(inducing)
        self.num_inducing = inducing.shape[0]

        # Initialize the mean value of the variational distribution
        q_mu = torch.tensor(
            np.zeros((num_inducing, output_dim)),
            dtype=self.dtype,
            device=self.device,
        )
        self.q_mu = torch.nn.Parameter(q_mu)

        # Initialize the variance of the variational distribution
        q_sqrt = np.tile(np.eye(num_inducing)[:, :, None], [1, 1, output_dim])
        li, lj = torch.tril_indices(num_inducing, num_inducing)
        triangular_q_sqrt = q_sqrt[li, lj]
        self.q_sqrt_tri = torch.tensor(
            triangular_q_sqrt,
            dtype=self.dtype,
            device=self.device,
        )
        self.q_sqrt_tri = torch.nn.Parameter(self.q_sqrt_tri)

        # Instantiate Standard Gaussian sampler
        self.gaussian_sampler = GaussianSampler(seed)

        # Instantiate the given kernel
        self.kernel = kernel

    def forward(self, inputs):
        """Creates a sample from the posterior distribution of the
        Gaussian process.

        Arguments
        ---------
        inputs : torch tensor of shape (..., M, D)
                 Batched input values

        Returns
        -------
        sample : torch tensor of shape (...., M, D_out)
                 Generated sample
        """
        # Compute K(u, u) and store it. shape (num_inducing, num_inducing)
        Ku = self.kernel(self.inducing, self.inducing)
        # Compute cholesky decomposition, lower triangular
        Lu = torch.linalg.cholesky(Ku + 1e-3 * torch.eye(Ku.shape[0]))

        # Shape (..., num_inducing, M)
        Kuf = self.kernel(self.inducing, inputs)
        # Shape (..., num_inducing, M)
        A = torch.cholesky_solve(Kuf, Lu)
        # Compute the posterior mean
        #  Shape (..., M, D_out)
        mean = A.transpose(-1, -2) @ self.q_mu

        # Shape (num_inducing, num_inducing, D_out, )
        q_sqrt = (
            torch.zeros(
                (self.num_inducing, self.num_inducing, self.output_dim)
            )
            .to(self.dtype)
            .to(self.device)
        )
        li, lj = torch.tril_indices(self.num_inducing, self.num_inducing)
        q_sqrt[li, lj] = self.q_sqrt_tri
        # Shape (num_inducing, num_inducing, D_out)
        SK = torch.einsum("ijd, kjd -> ikd", q_sqrt, q_sqrt)

        SK = torch.einsum("...im,ijd,...jm->...md", A, SK, A)

        # Create sample from posterior distribution
        z = self.gaussian_sampler(mean.shape[:-2])
        z = z.unsqueeze(-1).unsqueeze(-1)
        # sample = mu + Lz
        return mean + z * SK.sqrt()

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
        diag_indexes = np.cumsum(np.arange(1, self.num_inducing + 1)) - 1
        diag = self.q_sqrt_tri[diag_indexes]
        # Constant dimensionality term
        KL = -0.5 * self.output_dim * self.num_inducing

        # Log of determinant of covariance matrix.
        # Det(Sigma) = Det(q_sqrt q_sqrt^T) = Det(q_sqrt) Det(q_sqrt^T)
        # = prod(diag_s_sqrt)^2
        KL -= torch.sum(torch.log(torch.abs(diag)))

        # Trace term.
        KL += 0.5 * torch.sum(torch.square(self.q_sqrt_tri))

        # Mean term
        KL += 0.5 * torch.sum(torch.square(self.q_mu))

        return KL
