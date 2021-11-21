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
        return torch.normal(mean=0, std=1, size=size, generator=self.generator)


class GenerativeFunction(torch.nn.Module):
    def __init__(self,
                 noise_sampler,
                 num_samples,
                 num_outputs,
                 input_dim,
                 trainable,
                 seed,
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
        self.dtype = dtype

    def forward(self):
        """
        Generates the function samples.
        """
        raise NotImplementedError


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
        super().__init__(noise_sampler, num_samples, num_outputs, input_dim,
                         trainable, seed)

        generator = torch.Generator()
        generator.manual_seed(seed)

        def initializer(size):
            return torch.randn(size=size,
                               generator=generator,
                               dtype=self.dtype)

        self.initializer = initializer

        self.structure = structure
        self.w_mean_prior = w_mean_prior
        self.w_std_prior = w_std_prior
        self.b_mean_prior = b_mean_prior
        self.b_std_prior = b_std_prior
        self.trainable = trainable
        self.output_dim = num_outputs

        self.activation = activation

        dims = [input_dim] + self.structure + [self.output_dim]

        self.weights = []

        for i, (_in, _out) in enumerate(zip(dims, dims[1:])):

            w_mean_name = "w_mean_{}".format(i)
            w_std_name = "w_log_std_{}".format(i)
            b_mean_name = "b_mean_{}".format(i)
            b_std_name = "b_log_std_{}".format(i)

            self.register_parameter(
                name=w_mean_name,
                param=torch.nn.Parameter(
                    self.w_mean_prior +
                    0.01 * self.initializer(size=(_in, _out)),
                    requires_grad=True))
            self.register_parameter(
                name=w_std_name,
                param=torch.nn.Parameter(
                    self.w_std_prior + 4.6 +
                    0.0 * self.initializer(size=(_in, _out)),
                    requires_grad=True))
            self.register_parameter(
                name=b_mean_name,
                param=torch.nn.Parameter(self.b_mean_prior +
                                         0.1 * self.initializer(size=[_out]),
                                         requires_grad=True))
            self.register_parameter(
                name=b_std_name,
                param=torch.nn.Parameter(self.b_std_prior +
                                         0.1 * self.initializer(size=[_out]),
                                         requires_grad=True))

            self.weights.append((getattr(self, '{}'.format(w_mean_name)),
                                 getattr(self, '{}'.format(w_std_name)),
                                 getattr(self, '{}'.format(b_mean_name)),
                                 getattr(self, '{}'.format(b_std_name))))

    def forward(self, inputs):
        """
        Computes the output of the Bayesian neural network given the input
        values.
        """

        # Input has shape (N, D), we are replicating it self.num_samples
        # times in the first dimension (N, S, D)
        x = torch.unsqueeze(inputs, 1)
        x = torch.tile(x, (1, self.num_samples, 1))

        for (w_m, w_log_std, b_m, b_log_std) in self.weights[:-1]:
            # Get noise
            z_w = self.noise_sampler((self.num_samples, *w_log_std.shape))
            z_b = self.noise_sampler((self.num_samples, *b_log_std.shape))

            # Compute Gaussian samples
            w = z_w * torch.exp(w_log_std) + w_m
            b = z_b * torch.exp(b_log_std) + b_m
            x = self.activation(torch.einsum("nsi,sio->nso", x, w) + b)

        # Last layer has no activation function
        w_m, w_log_std, b_m, b_log_std = self.weights[-1]
        z_w = self.noise_sampler((self.num_samples, *w_log_std.shape))
        z_b = self.noise_sampler((self.num_samples, *b_log_std.shape))
        w = z_w * torch.exp(w_log_std) + w_m
        b = z_b * torch.exp(b_log_std) + b_m
        return torch.einsum("nsi,sio->nso", x, w) + b
