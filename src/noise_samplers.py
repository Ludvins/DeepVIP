

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
