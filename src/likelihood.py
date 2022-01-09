# Based on GPflow's likelihood

import numpy as np
import torch


class Likelihood(torch.nn.Module):
    def __init__(self, dtype=torch.float64, device=None):
        """"""
        super().__init__()
        self.dtype = dtype
        self.device = device

    def logdensity(self, x, mu, var):
        raise NotImplementedError

    def logp(self, F, Y):
        raise NotImplementedError

    def conditional_mean(self, F):
        raise NotImplementedError

    def conditional_variance(self, F):
        raise NotImplementedError

    def predict_mean_and_var(self, Fmu, Fvar):
        raise NotImplementedError

    def predict_logdensity(self, Fmu, Fvar, Y):
        raise NotImplementedError

    def variational_expectations(self, Fmu, Fvar, Y):
        raise NotImplementedError


class Gaussian(Likelihood):
    def __init__(self, log_variance=-5.0, dtype=torch.float64, device=None):
        """Gaussian Likelihood. Encapsulates the likelihood noise
        as a parameter.
        Arguments
        ---------
        log_variance : float of type self.dtype
                       Initial value for the logarithm of the variance.
        dtype : data-type
                Type of the log-variance parameter.
        device : torch device
                 Device in which to store the parameter.

        """
        super().__init__(dtype, device)
        # initialize parameter
        self.log_variance = torch.tensor(
            log_variance, dtype=dtype, device=self.device
        )
        self.log_variance = torch.nn.Parameter(self.log_variance)

    def logdensity(self, x, mu, var):
        """ Computes the log density of a one dimensional
        Gaussian distribution of mean mu and variance var, evaluated
        on x
        """
        return -0.5 * (np.log(2 * np.pi) + var.log() + (mu - x).square() / var)

    def logp(self, F, Y):
        """Computes the log likelihood of the targets Y under the predictions F,
        using the likelihoo variance."""
        return self.logdensity(Y, F, self.log_variance.exp())

    def predict_mean_and_var(self, Fmu, Fvar):
        """Returns the predictive mean and variance using the likelihood distribution.
        In this case, it results in accumulating the variances."""
        return Fmu, Fvar + self.log_variance.exp()

    def predict_density(self, Fmu, Fvar, Y):
        """Computes the predictive density of the targets given the means (Fmu) and
        vairances (Fvar)"""
        return self.logdensity(Y, Fmu, Fvar + self.log_variance.exp())

    def variational_expectations(self, Fmu, Fvar, Y):
        """Computes the variational expectation, i.e, the expectation under
        Q(f) ~ N(Fmu, Fvar) of the log likelihood P(y | f). As both distributions
        are Gaussian this can be computed in closed form."""
        return (
                -0.5 * np.log(2 * np.pi)
                - 0.5 * self.log_variance
                - 0.5 * ((Y - Fmu).square() + Fvar) / self.log_variance.exp()
            )
