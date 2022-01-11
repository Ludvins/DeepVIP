import numpy as np
import torch


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
    def __init__(self, seed=2147483647, device="cpu", dtype=torch.float64):
        """
        Generates noise samples from a Standar Gaussian distribution N(0, 1).

        Parameters:
        -----------
        seed : int
               Integer value used to generate reproducible results.
        device : torch.device
                 The device in which the computations are made.
        dtype : data-type
                The dtype of the layer's computations and weights.
        """
        super().__init__(seed, dtype)
        self.device = device
        self.generator = torch.Generator(device)
        self.generator.manual_seed(seed)

    def reset_seed(self):
        """
        Sets the random seed so that the same random samples can be
        generated if desired.
        """
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
            device=self.device,
        )


class UniformSampler(NoiseSampler):
    def __init__(self, seed=2147483647, device="cpu", dtype=torch.float64):
        """
        Generates noise samples from a continuous uniform distribution U[0,1].

        Parameters:
        -----------
        seed : int
               Integer value used to generate reproducible results.
        device : torch.device
                 The device in which the computations are made.
        dtype : data-type
                The dtype of the layer's computations and weights.
        """
        super().__init__(seed, dtype)
        self.device = device
        self.generator = torch.Generator(device)
        self.generator.manual_seed(seed)

    def reset_seed(self):
        """
        Sets the random seed so that the same random samples can be
        generated if desired.
        """
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
                  A sample from a Uniform distribution U(0, 1).
        """

        return torch.rand(
            size=size,
            generator=self.generator,
            dtype=self.dtype,
            device=self.device,
        )


class GenerativeFunction(torch.nn.Module):
    def __init__(
        self,
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
        """
        Generates the function samples.
        """
        raise NotImplementedError


class BayesLinear(GenerativeFunction):
    def __init__(
        self,
        input_dim,
        output_dim,
        w_mean,
        w_log_std,
        b_mean,
        b_log_std,
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
        input_dim : int
                    Dimensionality of the input values `x`.
        output_dim : int
                     Dimensionality of the function output.
        w_mean : torch tensor of shape (input_dim, output_dim)
                 Initial value w's mean.
        w_log_std : torch tensor of shape (input_dim, output_dim)
                    Logarithm of the initial standard deviation of w.
        b_mean : torch tensor of shape (1, output_dim)
                 Initial value b's mean.
        b_log_std : torch tensor of shape (1, output_dim)
                    Logarithm of the initial standard deviation of b.
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
            fix_random_noise=fix_random_noise,
            seed=seed,
            dtype=dtype,
        )

        self.zero_mean_prior = zero_mean_prior
        # Instantiate Standard Gaussian sampler
        self.gaussian_sampler = GaussianSampler(seed, device)

        # Check prior values fit the given dimensionality
        if (
            (w_mean.shape != (input_dim, output_dim))
            or (w_log_std.shape != (input_dim, output_dim))
            or (b_mean.size(0) != output_dim)
            or (b_log_std.size(0) != output_dim)
        ):
            raise RuntimeError(
                "Provided prior values do not fit the given" " dimensionality."
            )

        # Initialize parameters
        if zero_mean_prior:
            self.weight_mu = 0
            self.bias_mu = 0
        else:
            self.weight_mu = torch.nn.Parameter(w_mean)
            self.bias_mu = torch.nn.Parameter(b_mean)

        self.weight_log_sigma = torch.nn.Parameter(w_log_std)
        self.bias_log_sigma = torch.nn.Parameter(b_log_std)

    def forward(self, inputs, num_samples=None):
        """Forwards the given input through the Bayesian Neural Network.
        Generates as many samples of the stochastic output as indicated.

        Arguments
        ---------

        inputs : torch tensor of shape (S, N, D)
                 Input tensor where the last two dimensions are batch and
                 data dimensionality.

        num_samples : int
                      Number of samples to generate from the stochastic
                      function. If not specified, if inputs has at least
                      three dimensions, the first one will be used to
                      store the different samples. Otherwise, only one sample
                      is generated.

        """

        # Check the given input is valid
        if inputs.shape[-1] != self.input_dim:
            raise RuntimeError(
                "Input shape does not match stored data dimension"
            )

        if num_samples is None:
            # The number of samples corresponds to the first dimension
            if inputs.ndim >= 3:
                num_samples = inputs.size(0)
            else:
                num_samples = 1
                inputs = inputs.unsqueeze(0)

        # When generating the same random numbers, retrieve from stored
        #  noise dict when possible.
        if self.fix_random_noise:
            self.gaussian_sampler.reset_seed()

        z_w_shape = (num_samples, self.input_dim, self.output_dim)
        z_b_shape = (num_samples, 1, self.output_dim)

        # Generate Gaussian values
        z_w = self.gaussian_sampler(z_w_shape)
        z_b = self.gaussian_sampler(z_b_shape)

        # Perform reparameterization trick
        w = z_w * torch.exp(self.weight_log_sigma) + self.weight_mu
        b = z_b * torch.exp(self.bias_log_sigma) + self.bias_mu

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
        # Compute w's flattened mean and covariance diagonal matrix
        if self.zero_mean_prior:
            w_m = torch.zeros_like(self.weight_log_sigma)
            b_m = torch.zeros_like(self.bias_log_sigma)
        else:
            w_m = torch.flatten(self.weight_mu)
            b_m = torch.flatten(self.bias_mu)

        w_Sigma = torch.flatten(torch.square(torch.exp(self.weight_log_sigma)))
        b_Sigma = torch.flatten(torch.square(torch.exp(self.bias_log_sigma)))

        # Compute the KL divergence of w
        KL = -w_m.size(dim=0)
        KL += torch.sum(w_Sigma)
        KL += torch.sum(w_m ** 2)
        KL -= 2 * torch.sum(self.weight_log_sigma)

        # Compute the KL divergence of b
        KL -= b_m.size(dim=0)
        KL += torch.sum(b_Sigma)
        KL += torch.sum(b_m ** 2)
        KL -= 2 * torch.sum(self.bias_log_sigma)

        return KL / 2


class BayesianNN(GenerativeFunction):
    def __init__(
        self,
        structure,
        activation,
        input_dim=1,
        output_dim=1,
        dropout=0.0,
        seed=2147483647,
        fix_random_noise=False,
        zero_mean_prior=False,
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
        input_dim : int
                    Dimensionality of the input values `x`.
        output_dim : int
                     Dimensionality of the function output.
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
            fix_random_noise=fix_random_noise,
            seed=seed,
            dtype=dtype,
        )

        self.structure = structure
        self.activation = activation
        self.generator = torch.Generator()
        self.generator.manual_seed(self.seed)
        self.dropout = dropout
        # Create an array symbolizing the dimensionality of the data at
        # each inner layer.
        dims = [input_dim] + structure + [output_dim]
        layers = []

        gaussian_sampler = GaussianSampler(seed, device)

        # Loop over the input and output dimension of each sub-layer.
        for _in, _out in zip(dims, dims[1:]):
            w_mean = gaussian_sampler((_in, _out))
            w_log_std = torch.log(
                torch.abs(
                    gaussian_sampler((_in, _out))
                )
            )
            b_mean = gaussian_sampler([_out])
            b_log_std = torch.log(
                torch.abs(
                  gaussian_sampler([_out])
                )
            )

            layers.append(
                BayesLinear(
                    _in,
                    _out,
                    w_mean=w_mean.to(device),
                    w_log_std=w_log_std.to(device),
                    b_mean=b_mean.to(device),
                    b_log_std=b_log_std.to(device),
                    device=device,
                    fix_random_noise=fix_random_noise,
                    zero_mean_prior=zero_mean_prior,
                    seed=seed,
                    dtype=dtype,
                )
            )
        self.layers = torch.nn.ModuleList(layers)

    def forward(self, inputs, num_samples):
        """Forward pass over each inner layer, activation is applied on every
        but the last level.
        """

        # Replicate the input on the first dimension as many times as
        #  desired samples.
        x = torch.tile(
            inputs.unsqueeze(0),
            (num_samples, *np.ones(inputs.ndim, dtype=int)),
        )

        for layer in self.layers[:-1]:
            x = self.activation(layer(x))
            # Only perform dropout if in training mode
            if self.dropout > 0.0 and self.training:
                x = torch.nn.Dropout(self.dropout)(x)

        return self.layers[-1](x)

    def KL(self):
        """Computes the Kl divergence of the model as the addition of the
        KL divergences of its sub-models.
        """
        return torch.stack([layer.KL() for layer in self.layers]).sum()


class BNN_GP(GenerativeFunction):
    def __init__(
        self,
        input_dim=1,
        output_dim=1,
        inner_layer_dim=10,
        kernel_amp = 1.,
        kernel_length = 1.,
        seed=2147483647,
        fix_random_noise=False,
        device=None,
        dtype=torch.float64,
    ):
        """ Generates samples from a Bayesian Neural Network that
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
            fix_random_noise=fix_random_noise,
            seed=seed,
            dtype=dtype,
        )

        # Initialize variables and noise generators
        self.inner_layer_dim = inner_layer_dim
        self.generator = torch.Generator()
        self.gaussian_sampler = GaussianSampler(seed, device)
        self.uniform_sampler = UniformSampler(seed=seed, device=device)

        # Initialize parameters, logarithms are used in order to avoid
        #  constraining to positive values.
        self.log_kernel_amp = torch.nn.Parameter(torch.log(torch.tensor(kernel_amp, dtype = self.dtype)))
        self.log_kernel_length = torch.nn.Parameter(torch.log(torch.tensor(kernel_length, dtype = self.dtype)))
        
    def forward(self, inputs, num_samples):
        
        # Fix noise samples, i.e, fix prior samples
        if self.fix_random_noise:
            self.gaussian_sampler.reset_seed()
            self.uniform_sampler.reset_seed()

        # Sample noise values from Gaussian and uniform in order to
        # approximate the kernel
        z = self.gaussian_sampler(
            (self.input_dim, self.inner_layer_dim)
        )
        b = 2*np.pi * self.uniform_sampler(
            (1, self.inner_layer_dim)
        )

        # Compute kernel function, start by scaling the Gaussian noise by the kernel length
        w = z / torch.exp(self.log_kernel_length)
        # Compute the normalizing factor
        scale_factor = torch.sqrt(2 * torch.exp(self.log_kernel_amp) / self.inner_layer_dim)
        # Compute phi, shape [S, inner_dim]
        phi =  scale_factor * torch.cos(inputs @ w + b)

        # Once the kernel is approximated, generate the desired number
        # of samples from the appriximate GP.
        w = self.gaussian_sampler(
            (num_samples, self.inner_layer_dim, self.output_dim)
        )
        
        return phi @ w

