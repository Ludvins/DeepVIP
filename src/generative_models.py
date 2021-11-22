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
    def __init__(self, seed):
        """
        Generates noise samples from a Standar Gaussian distribution N(0, 1).

        Parameters:
        -----------
        seed : int
               Integer value used to generate reproducible results.

        """
        super().__init__(seed)
        self.generator = torch.Generator()  # TODO Check device
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
        return torch.normal(mean=0,
                            std=1,
                            size=size,
                            generator=self.generator,
                            dtype=torch.float64)


class GenerativeFunction(torch.nn.Module):
    def __init__(self,
                 noise_sampler,
                 input_dim,
                 output_dim,
                 dtype=torch.float64):
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
    def __init__(self,
                 noise_sampler,
                 input_dim,
                 output_dim,
                 w_mean_prior=0.0,
                 w_std_prior=-4.6,
                 b_mean_prior=0.01,
                 b_std_prior=0.0,
                 dtype=torch.float64):
        super(BayesLinear, self).__init__(noise_sampler,
                                          input_dim,
                                          output_dim,
                                          dtype=dtype)

        self.w_mean_prior = w_mean_prior
        self.w_std_prior = w_std_prior
        self.b_mean_prior = b_mean_prior
        self.b_std_prior = b_std_prior
        self.frozen = False

        self.weight_mu = torch.nn.Parameter(
            torch.tensor(
                self.noise_sampler((input_dim, output_dim)) * 0.01 +
                w_mean_prior))
        self.weight_log_sigma = torch.nn.Parameter(
            torch.tensor(
                self.noise_sampler((input_dim, output_dim)) * 0.01 +
                w_std_prior))
        self.bias_mu = torch.nn.Parameter(
            torch.tensor(
                self.noise_sampler([output_dim]) * 0.01 + b_mean_prior))
        self.bias_log_sigma = torch.nn.Parameter(
            torch.tensor(
                self.noise_sampler([output_dim]) * 0.01 + b_std_prior))

    def forward(self, inputs):

        assert inputs.shape[-1] == self.input_dim

        w = self.weight_mu + torch.exp(
            self.weight_log_sigma) * self.noise_sampler(
                [inputs.shape[0], self.input_dim, self.output_dim])
        b = self.bias_mu + torch.exp(self.bias_log_sigma) * self.noise_sampler(
            [inputs.shape[0], self.output_dim])
        return torch.einsum("ni, nio -> no", inputs, w) + b


class BayesianNN(GenerativeFunction):
    def __init__(
        self,
        noise_sampler,
        structure,
        activation,
        input_dim=1,
        output_dim=1,
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
        self.w_mean_prior = w_mean_prior
        self.w_std_prior = w_std_prior
        self.b_mean_prior = b_mean_prior
        self.b_std_prior = b_std_prior
        self.output_dim = output_dim

        self.activation = activation

        dims = [input_dim] + self.structure + [self.output_dim]

        layers = [
            BayesLinear(noise_sampler, _in, _out, w_mean_prior, w_std_prior,
                        b_mean_prior, b_std_prior)
            for _in, _out in zip(dims, dims[1:])
        ]
        self.layers = torch.nn.ModuleList(layers)

    def freeze(self):
        for layer in self.layers:
            layer.freeze()

    def defreeze(self):
        for layer in self.layers:
            layer.freeze()

    def forward(self, inputs):
        x = inputs
        for layer in self.layers[:-1]:
            x = self.activation(layer(x))

        return self.layers[-1](x)