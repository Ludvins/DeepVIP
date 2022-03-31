import numpy as np
import torch
from .noise_samplers import GaussianSampler, UniformSampler
from .noise_samplers import GaussianSamplerSobol, UniformSamplerSobol


class GenerativeFunction(torch.nn.Module):
    def __init__(
        self,
        num_samples,
        input_dim,
        output_dim,
        device=None,
        fix_random_noise=False,
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
        fix_random_noise : boolean
                           Wether to reset the Random Generator's seed in each
                           iteration.
        seed : int
               Initial seed for the random number generator.
        dtype : data-type
                The dtype of the layer's computations and weights.
        """
        super(GenerativeFunction, self).__init__()
        self.num_samples = num_samples
        self.output_dim = output_dim
        self.input_dim = input_dim
        self.fix_random_noise = fix_random_noise
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
        num_samples,
        input_dim,
        output_dim,
        device=None,
        fix_random_noise=False,
        zero_mean_prior=False,
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
            num_samples,
            input_dim,
            output_dim,
            device=device,
            fix_random_noise=fix_random_noise,
            seed=seed,
            dtype=dtype,
        )

        self.zero_mean_prior = zero_mean_prior
        # Instantiate Standard Gaussian sampler
        self.gaussian_sampler = GaussianSampler(seed, device)

        # If the BNN has zero mean, no parameters are considered for the
        # mean values the weights and bias variable
        if zero_mean_prior:
            self.weight_mu = 0
            self.bias_mu = 0
        else:
            self.weight_mu = torch.nn.Parameter(
                torch.zeros([input_dim, output_dim], dtype=dtype, device = device)
            )
            self.bias_mu = torch.nn.Parameter(torch.zeros([1, output_dim],
                                                          dtype=dtype,
                                                          device = device))

        self.weight_log_sigma = torch.nn.Parameter(
            torch.zeros([input_dim, output_dim], dtype=dtype, device = device)
        )
        self.bias_log_sigma = torch.nn.Parameter(
            torch.zeros([1, output_dim], dtype=dtype, device = device)
        )

        # Reset the generator's seed if fixed noise.
        self.gaussian_sampler.reset_seed()
        if self.fix_random_noise:
            self.noise = self.get_noise(first_call=True)

    def get_noise(self, first_call=False):
        if self.fix_random_noise and not first_call:
            return self.noise
        else:
            # Compute the shape of the noise to generate
            z_w_shape = (self.num_samples, self.input_dim, self.output_dim)
            z_b_shape = (self.num_samples, 1, self.output_dim)

            # Generate Gaussian values
            z_w = self.gaussian_sampler(z_w_shape)
            z_b = self.gaussian_sampler(z_b_shape)

            return (z_w, z_b)

    def forward(self, inputs):
        """Forwards the given input through the Bayesian Neural Network.
        Generates as many samples of the stochastic output as indicated.

        Arguments
        ---------

        inputs : torch tensor of shape (S, N, D)
                 Input tensor where the last two dimensions are batch and
                 data dimensionality.
        """

        # Check the given input is valid
        if inputs.shape[-1] != self.input_dim:
            raise RuntimeError("Input shape does not match stored data dimension")

        # Generate Gaussian values
        z_w, z_b = self.get_noise()

        # Perform reparameterization trick
        w = self.weight_mu + z_w * torch.exp(self.weight_log_sigma)
        b = self.bias_mu + z_b * torch.exp(self.bias_log_sigma)

        # Apply linear transformation.
        return inputs @ w + b

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
        num_samples,
        input_dim,
        output_dim,
        device=None,
        fix_random_noise=False,
        zero_mean_prior=False,
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
            num_samples,
            input_dim,
            output_dim,
            device=device,
            fix_random_noise=fix_random_noise,
            zero_mean_prior=zero_mean_prior,
            seed=seed,
            dtype=dtype,
        )
        # If the BNN has zero mean, no parameters are considered for the
        # mean values the weights and bias variable
        if zero_mean_prior:
            self.weight_mu = 0
            self.bias_mu = 0
        else:
            self.weight_mu = torch.nn.Parameter(torch.tensor(0.0))
            self.bias_mu = torch.nn.Parameter(torch.tensor(0.0))

        self.weight_log_sigma = torch.nn.Parameter(torch.tensor(0.0))
        self.bias_log_sigma = torch.nn.Parameter(torch.tensor(0.0))


class BayesianNN(GenerativeFunction):
    def __init__(
        self,
        structure,
        activation,
        num_samples,
        input_dim,
        output_dim,
        layer_model,
        dropout=0.0,
        seed=2147483647,
        fix_random_noise=True,
        zero_mean_prior=False,
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
            num_samples,
            input_dim,
            output_dim,
            device=device,
            fix_random_noise=fix_random_noise,
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
                    self.num_samples,
                    _in,
                    _out,
                    device=device,
                    fix_random_noise=fix_random_noise,
                    zero_mean_prior=zero_mean_prior,
                    seed=seed,
                    dtype=dtype,
                )
            )
        # Store the layers as ModuleList so that pytorch can handle
        # training/evaluation modes and parameters.
        self.layers = torch.nn.ModuleList(layers)

    def forward(self, inputs):
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
            (self.num_samples, *np.ones(inputs.ndim, dtype=int)),
        )

        for layer in self.layers[:-1]:
            # Apply BNN layer
            x = self.activation(layer(x))
            # Pytorch internally handles when the dropout layer is in
            # training mode. Moreover, if p = 0, no bernoully samples
            # are taken, so there is no additional computational cost
            # in calling this function in evaluation or p=0.
            x = self.dropout(x)

        # Last layer has identity activation function
        return self.layers[-1](x)

    def KL(self):
        """Computes the Kl divergence of the model as the addition of the
        KL divergences of its sub-models."""
        return torch.stack([layer.KL() for layer in self.layers]).sum()


class BayesianConvLayer(torch.nn.Module):
    def __init__(
        self,
        num_samples,
        in_channels,
        out_channels,
        kernel_size,
        seed=2147483647,
        fix_random_noise=False,
        device=None,
        dtype=torch.float64,
    ):
        super(BayesianConvLayer, self).__init__()

        self.num_samples = num_samples
        self.fix_random_noise = fix_random_noise
        self.gaussian_sampler = GaussianSampler(seed, device)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.device = device
        self.dtype = dtype

        self.kernel_size = (
            kernel_size
            if isinstance(kernel_size, tuple)
            else (kernel_size, kernel_size)
        )

        weights_shape = (self.out_channels, self.in_channels, *self.kernel_size)
        self.W_mu = torch.nn.Parameter(torch.zeros(weights_shape, device=self.device))
        #self.W_mu = torch.nn.Parameter(torch.tensor(0.0))
        self.W_log_std = torch.nn.Parameter(
            torch.zeros(weights_shape, device=self.device)
        )
        #self.W_log_std = torch.nn.Parameter(torch.tensor(0.0))
        self.bias_mu = torch.nn.Parameter(
            torch.zeros([self.out_channels], device=self.device)
        )
        #self.bias_mu = torch.nn.Parameter(torch.tensor(0.0))
        self.bias_log_std = torch.nn.Parameter(
            torch.zeros([self.out_channels], device=self.device)
        )
        #self.bias_log_std = torch.nn.Parameter(torch.tensor(0.0))

        # Reset the generator's seed if fixed noise.
        self.gaussian_sampler.reset_seed()
        if self.fix_random_noise:
            self.noise = self.get_noise(first_call=True)

    def get_noise(self, first_call=False):
        if self.fix_random_noise and not first_call:
            return self.noise
        else:
            # Compute the shape of the noise to generate
            z_w_shape = (self.num_samples, self.out_channels, self.in_channels, *self.kernel_size)
            z_b_shape = (self.num_samples, self.out_channels)

            # Generate Gaussian values
            z_w = self.gaussian_sampler(z_w_shape)
            z_b = self.gaussian_sampler(z_b_shape)

            return (z_w, z_b)

    def forward(self, input):

        # Generate Gaussian values
        z_w, z_b = self.get_noise()

        W = self.W_mu + z_w * torch.exp(self.W_log_std)
        b = self.bias_mu + z_b * torch.exp(self.bias_log_std)

        x = torch.stack(
            [
                torch.nn.functional.conv2d(input=x, weight=weight, bias=bias)
                for x, weight, bias in zip(input, W, b)
            ]
        )
        return x


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
        self.conv1 = BayesianConvLayer(num_samples, 1, 8, 3, device = device)
        self.conv2 = BayesianConvLayer(num_samples, 4, 16, 3, device = device)

        self.fc = BayesLinear(
            num_samples=20,
            input_dim=16*5*5,
            output_dim=output_dim,
            device=device,
            seed=0,
            dtype=dtype,
        )

    def forward(self, input):
        # in_channel dimension as 1
        N = input.shape[0]
        S = self.num_samples

        # Tile num samples and set input channels as 1
        x = input.reshape((input.shape[0], 1, *self.input_dim))
        x = torch.tile(
            x.unsqueeze(0),
            (self.num_samples, *np.ones(x.ndim, dtype=int)),
        )
        
        # First convolution
        x = self.conv1(x)
        x = self.activation(x)

        # MaxPooling 2D
        x = x.reshape((-1, *x.size()[2:]))
        x = torch.nn.functional.max_pool2d(x, 2)
        x = x.reshape((S, N, *x.size()[1:]))
        
        # Second convolution
        x = self.conv2(x)
        
        # MaxPooling 2D
        x = x.reshape((-1, *x.size()[2:]))
        x = torch.nn.functional.max_pool2d(x, 2)
        x = x.reshape((S, N, *x.size()[1:]))

        # Flatten
        x = x.reshape((S, N, -1))

        # Fully connected
        x = self.fc(x)

        return x



class BayesianConvNN2(GenerativeFunction):
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
        _in = 4
        _out = 2 * _in
        self.conv1 = torch.nn.Conv2d(1, _in, 3, device = device, dtype = dtype)
        self.conv2 = torch.nn.Conv2d(_in, _out, 3, device = device, dtype = dtype)

        self.fc = BayesLinear(
            num_samples=num_samples,
            input_dim=_out*5*5,
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
        

class BayesianLSTM(GenerativeFunction):
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
        h = 100
        self.lstm = torch.nn.LSTM(input_dim, h, dropout = 0,device = device, dtype = dtype)

        self.fc = BayesLinear(
            num_samples=20,
            input_dim=h,
            output_dim=output_dim,
            device=device,
            seed=0,
            dtype=dtype,
        )

    def forward(self, input):
        # in_channel dimension as 1
        N = input.shape[0]

        # Tile num samples and set input channels as 1
        x = input.reshape((N, 1, -1))
        # First convolution
        x = self.lstm(x)[0]
                
        # Flatten
        x = x.reshape((N, -1))
        x = torch.tile(
            x.unsqueeze(0),
            (self.num_samples, *np.ones(x.ndim, dtype=int)),
        )

        # Fully connected
        x = self.fc(x)

        return x


class GP(GenerativeFunction):
    def __init__(
        self,
        num_samples,
        input_dim=1,
        output_dim=1,
        inner_layer_dim=10,
        kernel_amp=1.0,
        kernel_length=1.0,
        seed=2147483647,
        fix_random_noise=False,
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
            num_samples,
            input_dim,
            output_dim,
            device=device,
            fix_random_noise=fix_random_noise,
            seed=seed,
            dtype=dtype,
        )

        # Initialize variables and noise generators
        self.inner_layer_dim = inner_layer_dim
        self.inner_layer_dim_inv = 1 / self.inner_layer_dim
        self.gaussian_sampler = GaussianSampler(seed)
        self.uniform_sampler = UniformSampler(seed)

        # Initialize parameters, logarithms are used in order to avoid
        #  constraining to positive values.
        self.log_kernel_amp = torch.nn.Parameter(
            torch.log(torch.tensor(kernel_amp, dtype=self.dtype))
        )
        self.log_kernel_length = torch.nn.Parameter(
            torch.log(torch.tensor(kernel_length, dtype=self.dtype))
        )

        # Sample noise values from Gaussian and uniform in order to
        # approximate the kernel
        self.z = self.gaussian_sampler((self.input_dim, self.inner_layer_dim))
        self.b = 2 * np.pi * self.uniform_sampler((1, self.inner_layer_dim))

        if self.fix_random_noise:
            self.noise = self.get_noise(first_call=True)

    def get_noise(self, first_call=False):
        if self.fix_random_noise and not first_call:
            return self.noise
        else:
            self.gaussian_sampler.reset_seed()
            # Compute the shape of the noise to generate
            w = self.gaussian_sampler((self.num_samples, self.inner_layer_dim, 1))

            return w

    def forward(self, inputs):
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

        # Compute kernel function, start by scaling the inputs by the kernel length
        x = inputs / torch.exp(self.log_kernel_length)
        # Compute the normalizing factor
        scale_factor = torch.sqrt(
            2.0 * torch.exp(self.log_kernel_amp) / self.inner_layer_dim_inv
        )
        # Compute phi, shape [N, inner_dim]
        phi = scale_factor * torch.cos(x @ self.z + self.b)

        # Once the kernel is approximated, generate the desired number
        # of samples from the approximate GP.
        w = self.get_noise()

        return phi @ w


class CosFunctions(GenerativeFunction):
    def __init__(
        self,
        num_samples,
        input_dim=1,
        output_dim=1,
        kernel_amp=1.0,
        kernel_length=1.0,
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

        # Initialize variables and noise generators
        self.gaussian_sampler = GaussianSampler(seed)
        self.uniform_sampler = UniformSampler(seed)

        # Initialize parameters, logarithms are used in order to avoid
        #  constraining to positive values.
        self.log_kernel_amp = torch.nn.Parameter(
            torch.log(torch.tensor(kernel_amp, dtype=self.dtype))
        )
        self.log_kernel_length = torch.nn.Parameter(
            torch.log(torch.tensor(kernel_length, dtype=self.dtype))
        )
        self.gaussian_sampler.reset_seed()
        if self.fix_random_noise:
            self.noise = self.get_noise(first_call=True)

    def get_noise(self, first_call=False):
        if self.fix_random_noise and not first_call:
            return self.noise
        else:
            # Sample noise values from Gaussian and uniform in order to
            # approximate the kernel
            w = self.gaussian_sampler((self.num_samples, self.input_dim, 1))
            b = 2 * np.pi * self.uniform_sampler((self.num_samples, 1, 1))
            return (w, b)

    def forward(self, inputs):
        """Computes aproximated samples of a Gaussian process prior
        with kernel RBF and mean 0.

        Parameters
        ----------
        inputs : torch tensor of shape (N, D)
                 Contains the minibatch of N points with dimensionality D.

        Returns
        -------
        samples : torch tensor of shape (num_samples, N, D)
                  All the samples from the approximated GP prior.
        """

        # Compute kernel function, start by scaling the inputs by the length
        x = inputs / torch.exp(self.log_kernel_length)

        # Compute the normalizing factor
        scale_factor = torch.sqrt(
            2.0 * torch.exp(self.log_kernel_amp) / self.output_dim
        )

        # Get noise values:
        # w comes from a standard Gaussian with shape
        #   (num_samples, input_dim, output_dim)
        # b comes from a uniform (0, 2pi) with shape
        #   (num_samples, 1, output_dim)
        w, b = self.get_noise()

        # Compute phi, shape (num_samples, batch_size, output_dim)
        phi = scale_factor * torch.cos(x @ w + b)

        return phi
