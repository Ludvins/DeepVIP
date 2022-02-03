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
        dtype : data-type
                The dtype of the layer's computations and weights.
        """
        self.seed = seed
        self.rng = np.random.default_rng(self.seed)

    def reset_seed(self):
        """
        Sets the random seed so that the same random samples can be
        generated if desired.
        """
        self.rng = np.random.default_rng(self.seed)

    def __call__(self):
        raise NotImplementedError


class GaussianSampler(NoiseSampler):
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

        return torch.tensor(self.rng.standard_normal(size=size))


class UniformSampler(NoiseSampler):
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

        return torch.tensor(self.rng.uniform(size=size))


class GaussianSamplerSobol():
    def __init__(self, seed):
        self.seed = seed
        
    def __call__(self, size):
        soboleng = torch.quasirandom.SobolEngine(dimension=size[1], seed= self.seed)
        draws = soboleng.draw(size[0], dtype = torch.float64) + 1e-6
        m = torch.distributions.normal.Normal(torch.tensor([0.0]), torch.tensor([1.0]))
        return torch.reshape(m.icdf(draws), shape = size)


class UniformSamplerSobol():
    def __init__(self, seed):
        self.seed = seed
        
    def __call__(self, size):
        soboleng = torch.quasirandom.SobolEngine(dimension=size[1], seed = self.seed)
        return torch.reshape(soboleng.draw(size[0], dtype = torch.float64), shape = size)