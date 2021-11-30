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
    def __init__(self, input_dim, output_dim, dtype=torch.float64):
        """
        Generates samples from a stochastic function using sampled
        noise values and input values.

        Parameters:
        -----------
        input_dim : int or array
                    Dimensionality of the input values `x`.
        num_outputs : int
                      Dimensionality of the function output.
        dtype : data-type
                The dtype of the layer's computations and weights.
        """
        super(GenerativeFunction, self).__init__()
        self.output_dim = output_dim
        self.input_dim = input_dim
        self.dtype = dtype

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
        super(BayesLinear, self).__init__(input_dim, output_dim, dtype=dtype)

        # Instantiate Standard Gaussian sampler
        self.gaussian_sampler = GaussianSampler()

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
                "Input shape does not match stored data dimension")

        # Given an input of shape (A1, A2, ..., AM, N, D), random values are
        # generated over every dimension but the batch one (N), that is,
        # z_w_shape is (A1, ..., AM, D, D_out) and z_b_shape is
        # (A1, ..., AM, 1, D_out)
        z_w_shape = (*inputs.shape[:-2], self.input_dim, self.output_dim)
        z_b_shape = (*inputs.shape[:-2], 1, self.output_dim)

        #self.gaussian_sampler.set_seed()

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
        # Compute w's flattened mean and covariance diagonal matrix
        w_m = torch.flatten(self.weight_mu)
        w_Sigma = torch.flatten(torch.square(torch.exp(self.weight_log_sigma)))
        w_m_prior = torch.flatten(self.w_mean_prior)
        w_Sigma_prior = torch.flatten(
            torch.square(torch.exp(self.w_log_std_prior)))

        # Compute b's flattened mean and covariance diagonal matrix
        b_m = torch.flatten(self.bias_mu)
        b_Sigma = torch.flatten(torch.square(torch.exp(self.bias_log_sigma)))
        b_m_prior = torch.flatten(self.b_mean_prior)
        b_Sigma_prior = torch.flatten(
            torch.square(torch.exp(self.b_log_std_prior)))

        # Compute the KL divergence of w
        KL = -w_m.size(dim=0)
        KL += torch.sum(w_Sigma / w_Sigma_prior)
        KL += torch.sum((w_m_prior - w_m)**2 / w_Sigma_prior)
        KL += 2 * torch.sum(self.w_log_std_prior) - 2 * torch.sum(
            self.weight_log_sigma)

        # Compute the KL divergence of b
        KL -= b_m.size(dim=0)
        KL += torch.sum(b_Sigma / b_Sigma_prior)
        KL += torch.sum((b_m_prior - b_m)**2 / b_Sigma_prior)
        KL += 2 * torch.sum(self.b_log_std_prior) - 2 * torch.sum(
            self.bias_log_sigma)

        return KL / 2


class BayesianNN(GenerativeFunction):
    def __init__(
        self,
        structure,
        activation,
        input_dim=1,
        output_dim=1,
        seed=0,
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
        super().__init__(input_dim, output_dim)

        self.seed = seed
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
                    w_mean_prior=torch.normal(mean=.0,
                                              std=.01,
                                              size=(_in, _out),
                                              generator=self.generator),
                    w_log_std_prior=torch.log(
                        torch.abs(
                            torch.normal(mean=.01,
                                         std=.0,
                                         size=(_in, _out),
                                         generator=self.generator))),
                    b_mean_prior=torch.normal(mean=.01,
                                              std=.1,
                                              size=[_out],
                                              generator=self.generator),
                    b_log_std_prior=torch.log(
                        torch.abs(
                            torch.normal(mean=.0,
                                         std=.1,
                                         size=[_out],
                                         generator=self.generator))),
                ))
        self.layers = torch.nn.ModuleList(layers)

    def forward(self, inputs):
        """ Forward pass over each inner layer, activation is applied on every
        but the last level.
        """
        x = inputs
        for layer in self.layers[:-1]:
            x = self.activation(layer(x))

        return self.layers[-1](x)

    def KL(self):
        """ Computes the Kl divergence of the model as the addition of the 
        KL divergences of its sub-models. 
        """
        return torch.stack([layer.KL() for layer in self.layers]).sum()
