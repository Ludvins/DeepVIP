import torch
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
    def __init__(self, device=None, seed=2147483647):
        """
        Generates noise samples from a Standar Gaussian distribution N(0, 1).

        Parameters:
        -----------
        seed : int
               Integer value used to generate reproducible results.

        """
        super().__init__(seed)
        self.generator = torch.Generator()
        self.generator.manual_seed(seed)

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
        return torch.normal(
            mean=0,
            std=1,
            size=size,
            generator=self.generator,
            dtype=torch.float64,
        )


class GenerativeFunction(torch.nn.Module):
    def __init__(
        self, noise_sampler, input_dim, output_dim, dtype=torch.float64
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
        """
        super(GenerativeFunction, self).__init__()
        self.noise_sampler = noise_sampler
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
        noise_sampler,
        input_dim,
        output_dim,
        w_mean_prior,
        w_log_std_prior,
        b_mean_prior,
        b_log_std_prior,
        dtype=torch.float64,
    ):
        super(BayesLinear, self).__init__(
            noise_sampler, input_dim, output_dim, dtype=dtype
        )

        self.w_mean_prior = w_mean_prior
        self.w_log_std_prior = w_log_std_prior
        self.b_mean_prior = b_mean_prior
        self.b_log_std_prior = b_log_std_prior

        self.weight_mu = torch.nn.Parameter(w_mean_prior)
        self.weight_log_sigma = torch.nn.Parameter(w_log_std_prior)
        self.bias_mu = torch.nn.Parameter(b_mean_prior)
        self.bias_log_sigma = torch.nn.Parameter(b_log_std_prior)

    def forward(self, inputs):

        assert inputs.shape[-1] == self.input_dim

        # inputs (A1, A2, ..., AM, N, D)
        # z_w_shape (A1, ..., AM, D, D_out)
        # z_b shape (A1, ..., AM, 1, D_out)
        z_w_shape = (*inputs.shape[:-2], self.input_dim, self.output_dim)
        z_b_shape = (*inputs.shape[:-2], 1, self.output_dim)

        z_w = self.noise_sampler(z_w_shape)
        z_b = self.noise_sampler(z_b_shape)

        w = self.weight_mu + torch.exp(self.weight_log_sigma) * z_w
        b = self.bias_mu + torch.exp(self.bias_log_sigma) * z_b
        return torch.einsum("...nd, ...do -> ...no", inputs, w) + b


class BayesianNN(GenerativeFunction):
    def __init__(
        self,
        noise_sampler,
        structure,
        activation,
        input_dim=1,
        output_dim=1,
    ):
        """
        Defines a Bayesian Neural Network.

        Parameters:
        -----------
        noise_sampler : NoiseSampler
                        Instance of NoiseSampler used to generate the noise
                        values.
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
        super().__init__(noise_sampler, input_dim, output_dim)

        self.structure = structure
        self.activation = activation

        dims = [input_dim] + self.structure + [self.output_dim]
        layers = []

        for _in, _out in zip(dims, dims[1:]):
            layers.append(
                BayesLinear(
                    noise_sampler,
                    _in,
                    _out,
                    w_mean_prior=torch.normal(
                        mean=0.0, std=0.01, size=(_in, _out)
                    ),
                    w_log_std_prior=4.6
                    + torch.log(
                        torch.normal(mean=0.01, std=0.0, size=(_in, _out))
                    ),
                    b_mean_prior=torch.normal(mean=0.01, std=0.1, size=[_out]),
                    b_log_std_prior=torch.normal(
                        mean=0.0, std=0.1, size=[_out]
                    ),
                )
            )
        self.layers = torch.nn.ModuleList(layers)

    def forward(self, inputs):
        x = inputs
        for layer in self.layers[:-1]:
            x = self.activation(layer(x))

        return self.layers[-1](x)
