import numpy as np
import torch

from src.quadrature import hermgauss, hermgaussquadrature
from .noise_samplers import GaussianSampler, UniformSampler


class GenerativeFunction(torch.nn.Module):
    def __init__(
        self,
        input_dim,
        output_dim,
        device=None,
        seed=2147483647,
        dtype=torch.float64,
    ):
        """
        Generates samples from a stochastic function using sampled
        noise values and input values.

        Parameters
        ----------
        num_samples : int
                      Number of samples to generate.
        input_dim : int
                    Dimensionality of the input values `x`.
        output_dim : int
                     Dimensionality of the function output.
        device : torch.device
                 The device in which the computations are made.
        seed : int
               Initial seed for the random number generator.
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
        """Makes the model parameters non-trainable."""
        for param in self.parameters():
            param.requires_grad = False

    def defreeze_parameters(self):
        """Set the model parameters as trainable."""
        for param in self.parameters():
            param.requires_grad = True

    def forward(self):
        raise NotImplementedError


class BayesLinear(GenerativeFunction):
    def __init__(
        self,
        input_dim,
        output_dim,
        device=None,
        fix_mean=False,
        fix_variance=False,
        seed=0,
        dtype=torch.float64,
    ):
        """
        Generates samples from a stochastic Bayesian Linear function
        f(x) = w^T x + b,   where w and b follow a Gaussian distribution,
        parameterized by their mean and log standard deviation.

        Parameters:
        -----------
        num_samples : int
                      Number of samples to generate.
        input_dim : int
                    Dimensionality of the input values `x`.
        output_dim : int
                     Dimensionality of the function output.
        device : torch.device
                 The device in which the computations are made.
        fix_random_noise : boolean
                           Wether to reset the Random Generator's seed in each
                           iteration.
        zero_mean_prior : boolean
                          wether to consider 0 mean prior or not, i. e, to
                          create variables for the mean values of the gaussian
                          distributions or fix these to 0.
        seed : int
               Initial seed for the random number generator.
        dtype : data-type
                The dtype of the layer's computations and weights.
        """
        super(BayesLinear, self).__init__(
            input_dim,
            output_dim,
            device=device,
            seed=seed,
            dtype=dtype,
        )

        self.fix_mean = fix_mean
        # Instantiate Standard Gaussian sampler
        self.gaussian_sampler = GaussianSampler(seed, device)

        # If the BNN has zero mean, no parameters are considered for the
        # mean values the weights and bias variable

        rng = np.random.default_rng(0)
        # input_dim = 1

        self.weight_mu = torch.nn.Parameter(
                    torch.tensor(
                        rng.normal(size = (input_dim, output_dim))*1,
                        dtype=dtype, 
                        device=device)
            )
        self.bias_mu = torch.nn.Parameter(
                torch.tensor(
                    rng.normal(size = (1, output_dim))*1, 
                    dtype=dtype,
                    device=device)
            )

        self.weight_log_sigma = torch.nn.Parameter(
            torch.tensor(
                rng.normal(size = (input_dim, output_dim))*0.1,
                dtype=dtype, 
                device=device)
        )
        self.bias_log_sigma = torch.nn.Parameter(
            torch.tensor(rng.normal(size = (1, output_dim)) * 0.1,
                         dtype=dtype, 
                         device=device)
        )

        if fix_mean:
            self.weight_mu.requires_grad = False
            self.bias_mu.requires_grad = False
        if fix_variance:
            self.weight_log_sigma.requires_grad = False
            self.bias_log_sigma.requires_grad = False

    def get_noise(self, num_samples):

        # Compute the shape of the noise to generate
        z_w_shape = (num_samples, self.input_dim, self.output_dim)
        z_b_shape = (num_samples, 1, self.output_dim)

        # Generate Gaussian values
        z_w = self.gaussian_sampler(z_w_shape)
        z_b = self.gaussian_sampler(z_b_shape)
        return (z_w, z_b)

    def forward(self, inputs, num_samples):
        """Forwards the given input through the Bayesian Neural Network.
        Generates as many samples of the stochastic output as indicated.

        Arguments
        ---------

        inputs : torch tensor of shape (S, N, D)
                 Input tensor where the last two dimensions are batch and
                 data dimensionality.
        """

        # Generate Gaussian values
        z_w, z_b = self.get_noise(num_samples)

        # Perform reparameterization trick
        w = self.weight_mu + z_w * torch.exp(self.weight_log_sigma)
        b = self.bias_mu + z_b * torch.exp(self.bias_log_sigma)

        # Apply linear transformation.
        return inputs @ w + b

    def forward_weights(self, inputs, w, b):
        # Apply linear transformation.
        return inputs @ w + b

    def forward_mean(self, inputs):
        """Forwards the given input through the Bayesian Neural Network.
        Generates as many samples of the stochastic output as indicated.

        Arguments
        ---------

        inputs : torch tensor of shape (S, N, D)
                 Input tensor where the last two dimensions are batch and
                 data dimensionality.
        """

        # Apply linear transformation.
        return inputs @ self.weight_mu + self.bias_mu

    def get_weights(self):
        return [self.weight_mu, self.bias_mu]

    def get_std_params(self):
        return torch.cat(
            [self.weight_log_sigma.flatten(), self.bias_log_sigma.flatten()], -1
        )

    def KL(self):
        """
        Computes the KL divergence of w and b to their prior distribution,
        a standard Gaussian N(0, I).

        Returns
        -------
        KL : int
             The addition of the 2 KL terms computed
        """

        # Compute covariance diagonal matrixes
        w_Sigma = torch.square(torch.exp(self.weight_log_sigma))
        b_Sigma = torch.square(torch.exp(self.bias_log_sigma))

        # Compute the 2*KL divergence of w
        KL = -self.input_dim * self.output_dim
        KL += torch.sum(w_Sigma)
        KL += torch.sum(self.weight_mu ** 2)
        KL -= 2 * torch.sum(self.weight_log_sigma)

        # Compute the 2*KL divergence of b
        KL -= self.output_dim
        KL += torch.sum(b_Sigma)
        KL += torch.sum(self.bias_mu ** 2)
        KL -= 2 * torch.sum(self.bias_log_sigma)

        # Re-escale
        return KL / 2


class SimplerBayesLinear(BayesLinear):
    def __init__(
        self,
        input_dim,
        output_dim,
        device=None,
        fix_mean=False,
        fix_variance = False,
        seed=0,
        dtype=torch.float64,
    ):
        """
        Generates samples from a stochastic Bayesian Linear function
        f(x) = w^T x + b,   where w and b follow a Gaussian distribution,
        parameterized by their mean and log standard deviation.

        Parameters:
        -----------
        num_samples : int
                      Number of samples to generate.
        input_dim : int
                    Dimensionality of the input values `x`.
        output_dim : int
                     Dimensionality of the function output.
        device : torch.device
                 The device in which the computations are made.
        fix_random_noise : boolean
                           Wether to reset the Random Generator's seed in each
                           iteration.
        zero_mean_prior : boolean
                          wether to consider 0 mean prior or not, i. e, to
                          create variables for the mean values of the gaussian
                          distributions or fix these to 0.
        seed : int
               Initial seed for the random number generator.
        dtype : data-type
                The dtype of the layer's computations and weights.
        """
        super(SimplerBayesLinear, self).__init__(
            input_dim,
            output_dim,
            device=device,
            fix_mean=fix_mean,
            seed=seed,
            dtype=dtype,
        )
        # If the BNN has zero mean, no parameters are considered for the
        # mean values the weights and bias variable
        rng = np.random.default_rng(0)

        self.weight_mu = torch.nn.Parameter(
                torch.tensor(
                    rng.normal(size = (input_dim, output_dim)),
                    dtype=dtype, 
                    device=device)
        )
        self.bias_mu = torch.nn.Parameter(
            torch.tensor(
                rng.normal(size = (1, output_dim)), 
                dtype=dtype,
                device=device)
        )



        self.weight_log_sigma = torch.nn.Parameter(
            torch.tensor(
                rng.normal(size = (input_dim, output_dim))*0.01,
                dtype=dtype, 
                device=device)
        )
        self.bias_log_sigma = torch.nn.Parameter(
            torch.tensor(rng.normal(size = (1, output_dim)) * 0.01,
                         dtype=dtype, 
                         device=device)
        )
        
        self.weight_mu_multi = torch.nn.Parameter(
                    torch.ones((1, 1),
                        dtype=dtype, 
                        device=device)
            )
        self.bias_mu_multi = torch.nn.Parameter(
                torch.ones((1, 1), 
                    dtype=dtype,
                    device=device)
            )

        self.weight_log_sigma_multi = torch.nn.Parameter(
            torch.ones((1, 1),
                dtype=dtype, 
                device=device)
        )
        self.bias_log_sigma_multi = torch.nn.Parameter(
            torch.ones((1, 1),
                         dtype=dtype, 
                         device=device)
        )
        
        self.weight_log_sigma.requires_grad = False
        self.bias_log_sigma.requires_grad = False
        self.weight_mu.requires_grad = False
        self.bias_mu.requires_grad = False
        
    def forward(self, inputs, num_samples):
        """Forwards the given input through the Bayesian Neural Network.
        Generates as many samples of the stochastic output as indicated.

        Arguments
        ---------

        inputs : torch tensor of shape (S, N, D)
                 Input tensor where the last two dimensions are batch and
                 data dimensionality.
        """

        # Generate Gaussian values
        z_w, z_b = self.get_noise(num_samples)

        # Perform reparameterization trick
        w = self.weight_mu * self.weight_mu_multi \
            + z_w * torch.exp(self.weight_log_sigma * self.weight_log_sigma_multi)
        b = self.bias_mu * self.bias_mu_multi \
            + z_b * torch.exp(self.bias_log_sigma * self.bias_log_sigma_multi)

        # Apply linear transformation.
        return inputs @ w + b

    def forward_weights(self, inputs, w, b):
        # Apply linear transformation.
        return inputs @ w + b

    def forward_mean(self, inputs):
        """Forwards the given input through the Bayesian Neural Network.
        Generates as many samples of the stochastic output as indicated.

        Arguments
        ---------

        inputs : torch tensor of shape (S, N, D)
                 Input tensor where the last two dimensions are batch and
                 data dimensionality.
        """

        # Apply linear transformation.
        return inputs @ (self.weight_mu * self.weight_mu_multi) \
            + self.bias_mu * self.bias_mu_multi

    def get_weights(self):
        return [self.weight_mu * self.weight_mu_multi,
                self.bias_mu * self.bias_mu_multi]

    def get_std_params(self):
        return torch.cat(
            [
                (self.weight_log_sigma * self.weight_log_sigma_multi).flatten(),
                (self.bias_log_sigma * self.bias_log_sigma_multi).flatten()
                ], -1
        )


class BayesianNN(GenerativeFunction):
    def __init__(
        self,
        structure,
        activation,
        input_dim,
        output_dim,
        layer_model,
        dropout=0.0,
        seed=2147483647,
        fix_mean=False,
        fix_variance = False,
        device=None,
        dtype=torch.float64,
    ):
        """
        Defines a Bayesian Neural Network with multiple layers.

        Parameters:
        -----------
        structure : array-like
                    Contains the inner dimensions of the Bayesian Neural
                    network. For example, [10, 10] symbolizes a Bayesian
                    network with 2 inner layers of width 10.
        activation : function
                     Activation function to use between inner layers.
        input_dim : int
                    Dimensionality of the input values `x`.
        output_dim : int
                     Dimensionality of the function output.
        layer_model :

        dropout : float between 0 and 1
                  The degree of dropout used after each activation layer
        device : torch.device
                 The device in which the computations are made.
        fix_random_noise : boolean
                           Wether to reset the Random Generator's seed in each
                           iteration.
        zero_mean_prior : boolean
                          Wether to consider zero mean layers.
        seed : int
               Initial seed for the random number generator.
        dtype : data-type
                The dtype of the layer's computations and weights.

        """
        super().__init__(
            input_dim,
            output_dim,
            device=device,
            seed=seed,
            dtype=dtype,
        )

        self.input_dim = input_dim

        # Store parameters
        self.structure = structure
        self.activation = activation
        self.generator = torch.Generator()
        self.generator.manual_seed(self.seed)
        self.dropout = torch.nn.Dropout(dropout)
        # Create an array symbolizing the dimensionality of the data at
        # each inner layer.
        dims = [self.input_dim] + structure + [output_dim]
        layers = []

        # Loop over the input and output dimension of each sub-layer.
        for _in, _out in zip(dims, dims[1:]):

            # Append the Bayesian linear layer to the array of layers
            layers.append(
                layer_model(
                    _in,
                    _out,
                    device=device,
                    fix_mean=fix_mean,
                    fix_variance = fix_variance,
                    seed=seed,
                    dtype=dtype,
                )
            )
        # Store the layers as ModuleList so that pytorch can handle
        # training/evaluation modes and parameters.
        self.layers = torch.nn.ModuleList(layers)

    def forward(self, inputs, num_samples):
        """Forward pass over each inner layer, activation is applied on every
        but the last level.

        Parameters
        ----------
        inputs : torch tensor of shape (N, D)
                 Contains the minibatch of N points with dimensionality D.

        Returns
        -------
        samples : torch tensor of shape (num_samples, N, D)
                  All the results of propagaring the input
                  num_samples times over the BNN.
        """

        # Replicate the input on the first dimension as many times as
        #  desired samples.

        x = torch.tile(
            inputs.unsqueeze(0),
            (num_samples, *np.ones(inputs.ndim, dtype=int)),
        )
        x = self.dropout(x)

        for layer in self.layers[:-1]:
            # Apply BNN layer
            x = self.activation(layer(x, num_samples))
            # Pytorch internally handles when the dropout layer is in
            # training mode. Moreover, if p = 0, no bernoully samples
            # are taken, so there is no additional computational cost
            # in calling this function in evaluation or p=0.
            x = self.dropout(x)

        # Last layer has identity activation function
        return self.layers[-1](x, num_samples)

    def forward_mean(self, inputs):
        x = inputs
        for layer in self.layers[:-1]:
            # Apply BNN layer
            x = self.activation(layer.forward_mean(x))
            # Pytorch internally handles when the dropout layer is in
            # training mode. Moreover, if p = 0, no bernoully samples
            # are taken, so there is no additional computational cost
            # in calling this function in evaluation or p=0.
            x = self.dropout(x)

        # Last layer has identity activation function
        return self.layers[-1].forward_mean(x)

    def forward_weights(self, inputs, weights):
        x = inputs
        for i, _ in enumerate(self.layers[:-1]):
            # Apply BNN layer
            x = self.activation(
                self.layers[i].forward_weights(x, weights[2 * i], weights[2 * i + 1])
            )
        # Last layer has identity activation function
        return self.layers[-1].forward_weights(x, weights[-2], weights[-1])

    def KL(self):
        """Computes the Kl divergence of the model as the addition of the
        KL divergences of its sub-models."""
        return torch.stack([layer.KL() for layer in self.layers]).sum()

    def get_weights(self):
        weights = []
        for layer in self.layers:
            weights = weights + layer.get_weights()
        return tuple(weights)

    def get_std_params(self):
        return torch.cat([layer.get_std_params() for layer in self.layers])


class Constant(torch.nn.Module):
    def __init__(self, seed, device, dtype):
        super().__init__()
        self.seed = seed
        self.dtype = dtype
        self.generator = torch.Generator()
        self.generator.manual_seed(self.seed)

        n = torch.poisson(3 * torch.ones(1), generator=self.generator)
        n = n.to(torch.int)
        self.locations = torch.sort(
            torch.cat(
                [
                    torch.rand(n, generator=self.generator, dtype=self.dtype),
                    torch.tensor([0.0, 1.0]),
                ]
            )
        )[0].reshape(-1, 1)
        self.values = torch.rand(
            size=[n + 2], generator=self.generator, dtype=self.dtype
        ).reshape(-1, 1)
        self.device = device

    def forward(self, x):
        a = (x.reshape(-1, 1) > self.locations.reshape(1, -1)).sum(axis=1) - 1
        return self.values[a]


class Linear(torch.nn.Module):
    def __init__(self, seed, device, dtype):
        super().__init__()
        self.seed = seed
        self.dtype = dtype
        self.generator = torch.Generator()
        self.generator.manual_seed(self.seed)

        n = torch.poisson(3 * torch.ones(1), generator=self.generator)
        n = n.to(torch.int)
        self.locations = torch.sort(
            torch.cat(
                [
                    torch.rand(n, generator=self.generator, dtype=self.dtype),
                    torch.tensor([0.0, 1.0]),
                ]
            )
        )[0].reshape(-1, 1)
        self.values = torch.rand(
            size=[n + 2], generator=self.generator, dtype=self.dtype
        ).reshape(-1, 1)
        self.device = device

    def forward(self, x):
        a = (x.reshape(-1, 1) > self.locations.reshape(1, -1)).sum(axis=1) - 1
        y = (self.values[a + 1] - self.values[a]) / (
            self.locations[a + 1] - self.locations[a]
        ) * (x - self.locations[a]) + self.values[a]
        return y


class PWConstant(GenerativeFunction):
    def __init__(
        self,
        num_samples,
        input_dim,
        output_dim,
        seed=2147483647,
        fix_random_noise=True,
        device=None,
        dtype=torch.float64,
    ):
        super().__init__(
            num_samples,
            input_dim,
            output_dim,
            device=device,
            fix_random_noise=fix_random_noise,
            seed=seed,
            dtype=dtype,
        )

        self.input_dim = input_dim

        self.functions = [
            Constant(self.seed - i, self.device, self.dtype)
            for i in range(self.num_samples)
        ]

    def forward(self, inputs):
        ret = [f(inputs) for f in self.functions]
        ret = torch.stack(ret)
        return ret


class PWLinear(GenerativeFunction):
    def __init__(
        self,
        num_samples,
        input_dim,
        output_dim,
        seed=2147483647,
        fix_random_noise=True,
        device=None,
        dtype=torch.float64,
    ):
        super().__init__(
            num_samples,
            input_dim,
            output_dim,
            device=device,
            fix_random_noise=fix_random_noise,
            seed=seed,
            dtype=dtype,
        )

        self.input_dim = input_dim

        self.functions = [
            Linear(self.seed - i, self.device, self.dtype)
            for i in range(self.num_samples)
        ]

    def forward(self, inputs):
        ret = [f(inputs) for f in self.functions]
        ret = torch.stack(ret)
        return ret


class BayesianConvNN(GenerativeFunction):
    def __init__(
        self,
        num_samples,
        input_dim,
        output_dim,
        activation,
        seed=2147483647,
        fix_random_noise=False,
        device=None,
        dtype=torch.float64,
    ):
        super().__init__(
            num_samples,
            input_dim,
            output_dim,
            device=device,
            fix_random_noise=fix_random_noise,
            seed=seed,
            dtype=dtype,
        )
        self.activation = activation
        self.channels = 32

        self.conv1 = torch.nn.Conv2d(1, self.channels, 3, device=device, dtype=dtype)
        self.conv2 = torch.nn.Conv2d(
            self.channels, self.channels * 2, 3, device=device, dtype=dtype
        )

        self.fc = BayesLinear(
            num_samples=num_samples,
            input_dim=self.channels * 2 * 5 * 5,
            output_dim=output_dim,
            device=device,
            seed=0,
            dtype=dtype,
        )

    def forward(self, input):
        # in_channel dimension as 1
        N = input.shape[0]

        # Tile num samples and set input channels as 1
        x = input.reshape((N, 1, *self.input_dim))
        # First convolution
        # Shape (N, 26, 26, 4)
        x = self.conv1(x)
        x = self.activation(x)

        # MaxPooling 2D
        # Shape (N, 13, 13, 4)
        x = torch.nn.functional.max_pool2d(x, 2)

        # Second convolution
        # Shape (N, 11, 11, 8)
        x = self.conv2(x)
        x = self.activation(x)

        # MaxPooling 2D
        # Shape (N, 5, 5, 8)
        x = torch.nn.functional.max_pool2d(x, 2)

        # Flatten
        x = x.reshape((N, -1))
        x = torch.tile(
            x.unsqueeze(0),
            (self.num_samples, *np.ones(x.ndim, dtype=int)),
        )

        # Fully connected
        return self.fc(x)


class BayesianTConvNN(GenerativeFunction):
    def __init__(
        self,
        num_samples,
        input_dim,
        output_dim,
        activation,
        seed=2147483647,
        fix_random_noise=False,
        device=None,
        dtype=torch.float64,
    ):
        super().__init__(
            num_samples,
            input_dim,
            output_dim,
            device=device,
            fix_random_noise=fix_random_noise,
            seed=seed,
            dtype=dtype,
        )
        self.activation = activation
        self.channels = 32

        self.fc = BayesLinear(
            num_samples=num_samples,
            input_dim=input_dim,
            output_dim=5 * 5 * self.channels * 2,
            device=device,
            seed=0,
            dtype=dtype,
        )

        self.m = torch.nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)

        self.deconv1 = torch.nn.ConvTranspose2d(
            self.channels * 2, self.channels, 4, device=device, dtype=dtype
        )
        self.deconv2 = torch.nn.ConvTranspose2d(
            self.channels, 1, 3, device=device, dtype=dtype
        )

    def forward(self, input):
        # Input shape (batch_size, inner_dim)
        N = input.shape[0]
        x = self.fc(input)
        x = self.activation(x)
        # Shape (num_samples, batch_size, 200)
        # 5 5 8
        x = x.reshape((-1, self.channels * 2, 5, 5))
        # 10 10 8
        x = self.m(x)
        # 13 13 8
        x = self.deconv1(x)
        x = self.activation(x)

        # MaxPooling 2D
        # 26 26 4
        x = self.m(x)
        # Second convolution
        # Shape (N, 11, 11, 8)
        x = self.deconv2(x)
        x = self.activation(x)

        # Flatten
        x = x.reshape((self.num_samples, N, *self.output_dim))
        # Fully connected
        return x


class GP(GenerativeFunction):
    def __init__(
        self,
        input_dim=1,
        output_dim=1,
        inner_layer_dim=10,
        kernel_amp=1,
        kernel_length=1,
        seed=2147483647,
        device=None,
        dtype=torch.float64,
    ):
        """Generates samples from a Bayesian Neural Network that
        approximates a GP with 0 mean and RBF kernel. More precisely,
        the RBF kernel is approximated by
        RBF(x1, x2) = E_w,b [(cos wx_1 + b)cos(wx_2 + b)]
        where w ~ N(0, 1) and b ~ U(0, 2pi). This implies that
        phi(x) = cos(wx + b)
        can be used as kernel function to aproximate the kernel
        RBF(x1, x2) ~ phi(x1) phi(x2)
        Samples from the process can be generated using the
        reparameterization trick, samples = phi * N(0, 1).
        Using this information, a network with two layers is used,
        the inner one computes phi and the last one the samples.
        Source: Random Features for Large-Scale Kernel Machines
                by Ali Rahimi and Ben Recht
        https://people.eecs.berkeley.edu/~brecht/papers/07.rah.rec.nips.pdf
        Parameters
        ----------
        input_dim : int
                    Dimensionality of the input data.
        output_dim : int
                     Dimensionality of the output targets.
        inner_layer_dim : int
                          Dimensionality of the hidden layer, i.e,
                          number of samples of w and b to aproximate
                          the expectation.
        kernel_amp : float
                     Amplitude of the RBF kernel that is being
                     approximated.
        kernel_length : float
                        Length of the RBF kernel that is being
                        approximated.
        seed : int
               Random number seed.
        fix_random_noise : Wether to fix the generated noise to be
                           the same on each call.
        device : torch device
                 Device in which computations are made.
        dtype : data dtype
                Data type to use (precision).
        """
        super().__init__(
            input_dim,
            output_dim,
            device=device,
            seed=seed,
            dtype=dtype,
        )

        # Initialize variables and noise generators
        self.inner_layer_dim = inner_layer_dim
        self.inner_layer_dim_inv = 1 / self.inner_layer_dim
        self.gaussian_sampler = GaussianSampler(seed, device)
        self.uniform_sampler = UniformSampler(seed, device)

        # Initialize parameters, logarithms are used in order to avoid
        #  constraining to positive values.
        self.log_kernel_amp = torch.nn.Parameter(
            torch.log(torch.tensor(kernel_amp, dtype=self.dtype))
        )
        self.log_kernel_length = torch.nn.Parameter(
            torch.log(torch.tensor(kernel_length, dtype=self.dtype))
        )
        self.rng = np.random.default_rng(0)
        self.z_mean = torch.tensor(
                 self.rng.normal(size = (self.input_dim,  self.inner_layer_dim)) * 1,
                 dtype=self.dtype, 
                 device=self.device)
        self.b_mean = torch.tensor(
                 self.rng.uniform(size = (1,  self.inner_layer_dim)) * 2 * np.pi,
                 dtype=self.dtype, 
                 device=self.device)
        self.w_mean = torch.tensor(
                self.rng.normal(size = (self.inner_layer_dim, output_dim)) * 0.01,
                dtype=self.dtype, 
                device=self.device)


    def get_noise(self, num_samples):

        z = self.z_mean + self.gaussian_sampler((self.input_dim, self.inner_layer_dim))
        b = self.b_mean +  2 * np.pi * self.uniform_sampler((1, self.inner_layer_dim)) - np.pi
        # Compute the shape of the noise to generate
        w = self.w_mean + self.gaussian_sampler((num_samples, self.inner_layer_dim, self.output_dim))

        return z, b, w

    def forward(self, inputs, num_samples):
        """Computes aproximated samples of a Gaussian process prior
        with kernel RBF and mean 0.
        Parameters
        ----------
        inputs : torch tensor of shape (N, D)
                 Contains the minibatch of N points with dimensionality D.
        num_samples: int
                     Number of samples to generate from the BNN.
                     Equivalently, inputs is propagated through the BNN
                     as many times as indicated by this variable. All
                     results are returned.
        Returns
        -------
        samples : torch tensor of shape (num_samples, N, D)
                  All the samples from the approximated GP prior.
        """
        x = inputs / torch.exp(self.log_kernel_length)
        scale_factor = torch.sqrt(
                2.0 * torch.exp(self.log_kernel_amp) / self.output_dim
            )
        z, b, w = self.get_noise(num_samples)

        phi = scale_factor * torch.cos(x @ z + b)

        return phi @ w
    
    def forward_mean(self, inputs):
        x = inputs / torch.exp(self.log_kernel_length)
        scale_factor = torch.sqrt(
                2.0 * torch.exp(self.log_kernel_amp) / self.output_dim
            )
        phi = scale_factor * torch.cos(x @ self.z_mean + self.b_mean)
        return phi @ self.w_mean

    def forward_weights(self, inputs, weights):
        x = inputs / torch.exp(self.log_kernel_length)
        scale_factor = torch.sqrt(
                2.0 * torch.exp(self.log_kernel_amp) / self.output_dim
            )
        phi = scale_factor * torch.cos(x @ weights[0] + weights[1])

        return phi @ weights[2]

    def get_weights(self):
        return tuple([self.z_mean, self.b_mean, self.w_mean])

    def get_std_params(self):
        z = torch.ones((self.input_dim, self.inner_layer_dim))
        b = torch.tensor(2*np.pi/(np.sqrt(12)),dtype = self.dtype) \
            + torch.zeros((1, self.inner_layer_dim))
        w = torch.ones((self.inner_layer_dim, self.output_dim))
        return torch.cat([torch.log(z.flatten()),
                          torch.log(b.flatten()),
                          torch.log(w.flatten())])
        
        
        
        
        
class GP(GenerativeFunction):
    def __init__(
        self,
        input_dim=1,
        output_dim=1,
        inner_layer_dim=10,
        kernel_amp=1,
        kernel_length=1,
        seed=2147483647,
        device=None,
        dtype=torch.float64,
    ):

        super().__init__(
            input_dim,
            output_dim,
            device=device,
            seed=seed,
            dtype=dtype,
        )

        # Initialize variables and noise generators
        self.inner_layer_dim = inner_layer_dim
        self.inner_layer_dim_inv = 1 / self.inner_layer_dim
        self.gaussian_sampler = GaussianSampler(seed, device)
        self.uniform_sampler = UniformSampler(seed, device)

        self.rng = np.random.default_rng(0)
        self.w1_mean = torch.tensor(
                 self.rng.normal(size = (self.input_dim,  self.inner_layer_dim)) * 1,
                 dtype=self.dtype, 
                 device=self.device)
        self.b1_mean = torch.tensor(
                 self.rng.normal(size = (1,  self.inner_layer_dim)) * 0.1,
                 dtype=self.dtype, 
                 device=self.device)
        self.w2_mean = torch.tensor(
                self.rng.normal(size = (self.inner_layer_dim, output_dim)) * 0.01,
                dtype=self.dtype, 
                device=self.device)
        self.b2_mean = torch.tensor(
                 self.rng.normal(size = (1,  output_dim)) * 0.1,
                 dtype=self.dtype, 
                 device=self.device)

        self.activation = lambda x:  0.5 * (1 + torch.erf((x) / torch.sqrt(torch.tensor(2))))

    def get_noise(self, num_samples):

        w1 = self.w1_mean + self.gaussian_sampler((num_samples, self.input_dim, self.inner_layer_dim)).to(self.dtype)
        b1 = self.b1_mean + self.gaussian_sampler((num_samples, 1, self.inner_layer_dim)).to(self.dtype)
        w2 = self.w2_mean + self.gaussian_sampler((num_samples, self.inner_layer_dim, self.output_dim)).to(self.dtype)
        b2 = self.b2_mean + self.gaussian_sampler((num_samples, 1, self.output_dim)).to(self.dtype)

        return w1, b1, w2, b2

    def forward(self, inputs, num_samples):
        """Computes aproximated samples of a Gaussian process prior
        with kernel RBF and mean 0.
        Parameters
        ----------
        inputs : torch tensor of shape (N, D)
                 Contains the minibatch of N points with dimensionality D.
        num_samples: int
                     Number of samples to generate from the BNN.
                     Equivalently, inputs is propagated through the BNN
                     as many times as indicated by this variable. All
                     results are returned.
        Returns
        -------
        samples : torch tensor of shape (num_samples, N, D)
                  All the samples from the approximated GP prior.
        """
        x = inputs
            
        w1, b1, w2, b2 = self.get_noise(num_samples)

        phi = self.activation(x @ w1 + b1) / np.sqrt(self.inner_layer_dim)

        return phi @ w2 + b2
    
    def forward_mean(self, inputs):
        x = inputs  
        x = x @ self.w1_mean + self.b1_mean

        phi = self.activation(x)  / np.sqrt(self.inner_layer_dim)
        return phi @ self.w2_mean + self.b2_mean

    def forward_weights(self, inputs, weights):
        x = inputs
            
        phi = self.activation(x @ weights[0] + weights[1])  / np.sqrt(self.inner_layer_dim)

        return phi @ weights[2] + weights[3]
    
    def GP_mean(self, inputs):
        #return self.forward_mean(inputs)
        x = inputs / torch.exp(self.log_kernel_length)
        scale_factor = torch.exp(self.log_kernel_amp)
            
        aux1 = x @ self.w1_mean + self.b1_mean
        aux2 = torch.sqrt(1 + torch.sum(x*x, 1) ).unsqueeze(-1)
        x = aux1 / aux2
        
        phi = scale_factor * self.activation(x)

        return phi @ self.w2_mean + self.b2_mean
    
    # def GP_cov(self, x1, x2):
        
    #     x1 = x1 / torch.exp(self.log_kernel_length)
    #     x2 = x2 / torch.exp(self.log_kernel_length)

    #     scale_factor = torch.sqrt(
    #             2.0 * torch.exp(self.log_kernel_amp) / self.output_dim
    #         )

    #     mean_v1 = (x1 @ self.w1_mean + self.b1_mean).squeeze(0)
    #     mean_v2 = (x2 @ self.w1_mean + self.b1_mean).squeeze(0)

    #     var_v1 = torch.sum(x1**2) * torch.ones(self.inner_layer_dim)
    #     var_v2 = torch.sum(x2**2) * torch.ones(self.inner_layer_dim)
    #     cov_v1v2 = torch.sum(x1*x2) * torch.ones(self.inner_layer_dim)
    #     f = lambda a: self.activation(a) \
    #         * self.activation(
    #             (mean_v2 + cov_v1v2 / var_v1  * (a - mean_v1)) 
    #             / 
    #             (torch.sqrt(1 + var_v2 - cov_v1v2**2 / var_v1))
    #         )
        
    #     xn, wn = hermgauss(20, self.dtype, self.device)
    #     gh_x = xn.unsqueeze(-1)
    #     # Shape (N, num_hermite)
    #     Xall = gh_x * torch.sqrt(torch.clip(2.0 * var_v1, min=1e-10)) + mean_v1
    #     # Shape (num_hermite, 1)
    #     gh_w = wn.reshape(-1, 1) / torch.sqrt(torch.tensor(np.pi))
    #     feval = f(Xall)
    #     # Shape (N, num_hermite)
    #     feval = feval.T @ gh_w

    #     a =  1 + scale_factor * torch.sum(feval / (self.inner_layer_dim + self.w2_mean**2)) - self.GP_mean(x1.unsqueeze(-1))*self.GP_mean(x2.unsqueeze(-1))
    #     return a
    
    
    def GP_cov(self, x1, x2 = None):
        
        if x2 is None:
            x2 = x1
        
        # Shape (N1, D1)
        x1 = x1 / torch.exp(self.log_kernel_length)
        # Shape (N2, D1)
        x2 = x2 / torch.exp(self.log_kernel_length)

        scale_factor = torch.exp(self.log_kernel_amp)

        # Shape (N1, 1, D2)
        mean_v1 = (x1 @ self.w1_mean + self.b1_mean).unsqueeze(1)
        # Shape (1, N2, D2)
        mean_v2 = (x2 @ self.w1_mean + self.b1_mean).unsqueeze(0)
        

        # Shape (N1, 1, D2)        
        var_v1 = torch.tile(torch.sum(x1 * x1, dim = -1, keepdim=True).unsqueeze(1), (1, 1, self.inner_layer_dim))
        # Shape (1, N2, D2)
        var_v2 = torch.tile(torch.sum(x2 * x2, dim = -1, keepdim=True).unsqueeze(0) , (1, 1,self.inner_layer_dim))
        # Shape (N1, N2, D2)
        cov_v1v2 = torch.tile((x1 @ x2.T).unsqueeze(-1) , (1,1,self.inner_layer_dim))

        def f(a):
            return self.activation(a) \
            * self.activation(
                (mean_v2 + cov_v1v2 / var_v1  * (a - mean_v1)) 
                / 
                (torch.sqrt(1 + var_v2 - cov_v1v2**2 / var_v1))
            )
        
        xn, wn = hermgauss(200, self.dtype, self.device)

        gh_x = xn.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        # Shape (N, num_hermite)
        
        # Shape (Num hermite, N, Dim)
        Xall = gh_x * torch.sqrt(torch.clip(2.0 * var_v1, min=1e-10)) + mean_v1

        # Shape (num_hermite, 1, 1)
        gh_w = wn.reshape(-1, 1, 1, 1) / torch.sqrt(torch.tensor(np.pi))
        feval = f(Xall)

        # Shape (N, num_hermite)
        feval = torch.sum(feval * gh_w, 0)

        # E[ (f(x) - m(x)) (f(x') - m(x') ] =
        #   E[ f(x) * f(x') + f(x) * m(x') + f(x') * m(x') + m(x) * m(x')]
        
        a= 1 +  torch.sum( scale_factor * feval / self.inner_layer_dim, -1) \
            - self.GP_mean(x1) @ self.GP_mean(x2).T
        return a
    
    
    
    def get_weights(self):
        return tuple([self.w1_mean, self.b1_mean, self.w2_mean, self.b2_mean])

    def get_std_params(self):
        w1 = torch.ones((self.input_dim, self.inner_layer_dim))
        b1 = torch.ones((1, self.inner_layer_dim))
        b2 = torch.ones((1, self.output_dim))
        w2 = torch.ones((self.inner_layer_dim, self.output_dim)) / np.sqrt(self.inner_layer_dim)
        return torch.cat([torch.log(w1.flatten()),
                          torch.log(b1.flatten()),
                          torch.log(w2.flatten()),
                          torch.log(b2.flatten()),
                          ])
        
        