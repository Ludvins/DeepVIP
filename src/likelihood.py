# Based on GPflow's likelihood

from itertools import product
import numpy as np
import torch
from torch.nn.functional import one_hot


class Likelihood(torch.nn.Module):
    def __init__(self, dtype=torch.float64, device=None):
        """"""
        super().__init__()
        self.dtype = dtype
        self.device = device

    def logdensity(self, x, mu, var):
        """
        Given a Normal distribution for the latent function, and a datum Y,
        compute the (log) predictive density of Y.

        i.e. if
            q(f) = N(Fmu, Fvar)

        and this object represents

            p(y|f)

        then this method computes the predictive density

           \int p(y=Y|f)q(f) df
        """
        raise NotImplementedError("implement the logdensity function\
                                  for this likelihood")
    def logp(self, F, Y):
        """
        Return the log density of the data given the function values.
        """
        raise NotImplementedError("implement the logp function\
                                  for this likelihood")

    def conditional_mean(self, F):
        """
        Given a value of the latent function, compute the mean of the data

        If this object represents

            p(y|f)

        then this method computes

            \int y p(y|f) dy
        """
        raise NotImplementedError

    def conditional_variance(self, F):
        """
        Given a value of the latent function, compute the variance of the data

        If this object represents

            p(y|f)

        then this method computes

            \int y^2 p(y|f) dy  - [\int y p(y|f) dy] ^ 2

        """
        raise NotImplementedError
    
    
    def predict_mean_and_var(self, Fmu, Fvar):
        """
        Given a Normal distribution for the latent function,
        return the mean of Y

        if
            q(f) = N(Fmu, Fvar)

        and this object represents

            p(y|f)

        then this method computes the predictive mean

           \int\int y p(y|f)q(f) df dy

        and the predictive variance

           \int\int y^2 p(y|f)q(f) df dy  - [ \int\int y^2 p(y|f)q(f) df dy ]^2
        """
    def predict_logdensity(self, Fmu, Fvar, Y):
        """
        Given a Normal distribution for the latent function, and a datum Y,
        compute the (log) predictive density of Y.

        i.e. if
            q(f) = N(Fmu, Fvar)

        and this object represents

            p(y|f)

        then this method computes the predictive density

           \int p(y=Y|f)q(f) df

        """
        raise NotImplementedError

    def variational_expectations(self, Fmu, Fvar, Y):
        """
        Compute the expected log density of the data, given a Gaussian
        distribution for the function values.

        if
            q(f) = N(Fmu, Fvar)

        and this object represents

            p(y|f)

        then this method computes

           \int (\log p(y|f)) q(f) df.

        """
        raise NotImplementedError


class Gaussian(Likelihood):
    def __init__(
        self, log_variance=-5.0, dtype=torch.float64, device=None
    ):
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
        self.log_variance = torch.tensor(log_variance, dtype=dtype, device=self.device)
        self.log_variance = torch.nn.Parameter(self.log_variance)

    def logdensity(self, x, mu, var):
        """Computes the log density of a one dimensional
        Gaussian distribution of mean mu and variance var, evaluated
        on x
        """
        return -0.5 * (np.log(2 * np.pi) + var.log() + (mu - x).square() / var)

    def logp(self, F, Y):
        """Computes the log likelihood of the targets Y under the predictions F,
        using the likelihood variance."""
        return self.logdensity(Y, F, self.log_variance.exp())

    def predict_mean_and_var(self, Fmu, Fvar):
        """Returns the predictive mean and variance using the likelihood distribution.
        In this case, it results in accumulating the variances."""
        return Fmu, Fvar + self.log_variance.exp()

    def predict_logdensity(self, Fmu, Fvar, Y):
        """Computes the predictive density of the targets given the means (Fmu) and
        variances (Fvar)"""
        return self.logdensity(Y, Fmu, Fvar + self.log_variance.exp())

    def variational_expectations(self, Fmu, Fvar, Y, alpha):
        """Computes the variational expectation, i.e, the expectation under
        Q(f) ~ N(Fmu, Fvar) of the log likelihood P(y | f).

        As both distributions are Gaussian this can be computed in closed form.
        """

        if alpha == 0:
            logpdf = (
                -0.5 * np.log(2 * np.pi)
                - 0.5 * self.log_variance
                - 0.5 * ((Y - Fmu).square() + Fvar) / self.log_variance.exp()
            )
            return torch.mean(logpdf, dim=0)

        # Black-box alpha-energy
        # Number of predictive mixtures
        S = torch.tensor(Fmu.shape[0])

        variance = torch.exp(self.log_variance)
        # Proportionality constant
        C = (
            torch.sqrt(2 * torch.pi * variance / alpha)
            / torch.sqrt(2 * torch.pi * variance) ** alpha
        )

        logpdf = self.logdensity(Y, Fmu, Fvar + variance / alpha)
        logpdf = torch.logsumexp(logpdf, dim=0) + torch.log(C) - torch.log(S)
        return logpdf / alpha



class MultiClass(Likelihood):
    def __init__(self, num_classes, dtype, device, epsilon = 0.001):
        super().__init__(dtype, device)
        self.num_gauss_hermite_points = 20
        self.num_classes = num_classes
        self.epsilon = torch.tensor(epsilon)
        self.K1 = self.epsilon/(self.num_classes - 1)
        
    def logdensity(self, mean, var, x):
        """ Computes the probability of the labels (x) comming from a Bernoulli with
        mean (mean) and variance (var).
        """
        # Set labels to one-hot notation
        oh_on = one_hot(x.long().flatten(), self.num_classes).type(self.dtype)
        # One_hot multiplied by the probability of each label 
        #  contains only the probability of the given targets.
        return torch.log(torch.sum(oh_on * mean, -1))
    
    def logp(self, F, Y):
        """
        Return the log density of the data given the function values.
        
        Parameters
        ----------
        F : torch tensor of shape (num_data, n_classes)
            Contains the score of each class for each data point on each
            mixture.
        Y : torch tensor of shape (num_data, 1)
            Contains the true class for each data sample.
        
        """
        # Tensor of shape (num_data, 1), with 1 if the prediction is correct
        # and 0 otherwise.
        hits = torch.eq(torch.unsqueeze(torch.argmax(F, -1), -1), Y.type(torch.int))
        yes = torch.ones_like(Y, dtype=self.dtype) - self.epsilon
        no = torch.zeros_like(Y, dtype=self.dtype) + self.K1
        p = torch.where(hits, yes, no)
        return torch.log(p)

    def conditional_mean(self, F):
        """
        Given a value of the latent function, compute the probability of the data.
        
        Parameters
        ----------
        F : torch tensor of shape (num_data, n_classes)
            Contains the score of each class for each data point on each
            mixture.
        
        Returns
        -------
        oh : torch tensor of shape (num_data, n_classes)
             Contains the probability of the data.
        """
        # Compute predictions as maximal scores
        predictions = torch.argmax(F, -1)
        # Compute One-hot representation
        oh = one_hot(predictions, self.num_classes)
        # Probability of ones is 1-epsilon
        oh[oh == 1] -= self.epsilon
        # Probability of zeros is K1
        oh[oh == 0] += self.K1
        return oh

    def conditional_variance(self, F):
        p = self.conditional_mean(F)
        return p - torch.square(p)

    def predict_mean_and_var(self, Fmu, Fvar):
        # Size (n_classes, n_data)
        possible_outputs = [torch.full([Fmu.size()[0], 1],i) for i in
                                range(self.num_classes)]
        # Size (n_classes, n_data)
        ps = [self.predict_density(Fmu, Fvar, po) for po in possible_outputs]
        # Shape (n_data, n_classes)
        ps = torch.concat(ps, dim = 1)
        return ps, ps - torch.square(ps)
    
    def hermgauss(self, n):
        # Return the locations and weights of GH quadrature
        x, w = np.polynomial.hermite.hermgauss(n)
        return torch.tensor(x, dtype = self.dtype),  torch.tensor(w, dtype = self.dtype)

    def prob_is_largest(self, Y, mu, var, gh_x, gh_w):
        # Work out what the mean and variance is of the indicated latent function.
        # Shape (num_samples, num_classes)
        oh_on = one_hot(Y.long().flatten(), self.num_classes).type(self.dtype)
        # Only mean and var values corresponging to true label remain. The rest are
        #  multiplied by 0. In short, that summation equals to retrieve the 
        #  mean and var on the true label for each sample
        # Shape (num_samples, 1)
        mu_selected = torch.sum(oh_on * mu, 1).reshape(-1, 1)
        var_selected = torch.sum(oh_on * var, 1).reshape(-1, 1)

        var_selected = torch.sqrt(torch.clip(2. * var_selected, min = 1e-10))
        
        # Generate Gauss Hermite grid. Shape (num_samples, num_hermite)
        X = mu_selected + gh_x * var_selected
        # Compute the CDF of the Gaussian between the latent functions and the grid,
        # (including the selected function).
        # Shape (num_samples, num_classes, num_hermite)
        dist = (torch.unsqueeze(X, 1) - torch.unsqueeze(mu, 2)) / torch.unsqueeze(var_selected, 2)
        cdfs = 0.5 * (1.0 + torch.erf(dist / np.sqrt(2.0)))
        cdfs = cdfs * (1 - 2e-4) + 1e-4

        # Blank out all the distances on the selected latent function.
        # Get off values as abs(oh_on - 1)
        oh_off = torch.abs(oh_on - 1)
        cdfs = cdfs * torch.unsqueeze(oh_off, 2) + torch.unsqueeze(oh_on, 2)

        # Take the product over the latent functions, and the sum over the GH grid.
        gh_w = gh_w.unsqueeze(-1)/ np.sqrt(np.pi)
        return torch.prod(cdfs, dim = 1) @ gh_w

    def predict_density(self, Fmu, Fvar, Y):
        gh_x, gh_w = self.hermgauss(self.num_gauss_hermite_points)
        p = self.prob_is_largest(Y, Fmu, Fvar, gh_x, gh_w)
        return  p * (1 - self.epsilon) + (1. - p) * (self.K1)
        
    def predict_logdensity(self, Fmu, Fvar, Y):
        return self.log(self.predict_density(Fmu, Fvar, Y))

    def variational_expectations(self, Fmu, Fvar, Y, alpha):
        """
        Compute the expected log density of the data, given a Gaussian
        distribution for the function values.

        if
            q(f) = N(Fmu, Fvar)

        and this object represents

            p(y|f)

        then this method computes

           \int (\log p(y|f)) q(f) df.


        Here, we implement a Gauss-Hermite quadrature routine.
        """
        gh_x, gh_w = self.hermgauss(self.num_gauss_hermite_points)
        p = self.prob_is_largest(Y, Fmu, Fvar, gh_x, gh_w)
        ve = p * torch.log(1.0 - self.epsilon) + (1.0 - p) * torch.log(
            self.K1
        )
        return torch.sum(ve, dim=-1)




class BroadcastedLikelihood(Likelihood):
    def __init__(self, likelihood):
        super().__init__(likelihood.dtype, likelihood.device)
        self.likelihood = likelihood

        
    def _broadcast(self, f, vars_SND, vars_ND):
        S, N, D = vars_SND[0].size()
        vars_tiled = [torch.tile(x[None, :, :], [S, 1, 1]) for x in vars_ND]

        flattened_SND = [torch.reshape(x, [S*N, D]) for x in vars_SND]
        flattened_tiled = [torch.reshape(x, [S*N, -1]) for x in vars_tiled]

        flattened_result = f(flattened_SND, flattened_tiled)
        if isinstance(flattened_result, tuple):
            return [torch.reshape(x, [S, N, -1]) for x in flattened_result]
        else:
            return torch.reshape(flattened_result, [S, N, -1])

    def logdensity(self, p, q, x):
        f = lambda vars_SND, vars_ND: self.likelihood.logdensity(vars_SND[0],
                                                                vars_SND[1],
                                                                vars_ND[0],
                                                                )
        return self._broadcast(f,[p, q], [x])
    
    def variational_expectations(self, Fmu, Fvar, Y, alpha):
        f = lambda vars_SND, vars_ND: self.likelihood.variational_expectations(vars_SND[0],
                                                                                vars_SND[1],
                                                                                vars_ND[0],
                                                                                alpha)
        return self._broadcast(f,[Fmu, Fvar], [Y])

    def logp(self, F, Y):
        f = lambda vars_SND, vars_ND: self.likelihood.logp(vars_SND[0], vars_ND[0])
        return self._broadcast(f, [F], [Y])

    def conditional_mean(self, F):
         f = lambda vars_SND, vars_ND: self.likelihood.conditional_mean(vars_SND[0])
         return self._broadcast(f,[F], [])

    def conditional_variance(self, F):
         f = lambda vars_SND, vars_ND: self.likelihood.conditional_variance(vars_SND[0])
         return self._broadcast(f,[F], [])

    def predict_mean_and_var(self, Fmu, Fvar):
         f = lambda vars_SND, vars_ND: self.likelihood.predict_mean_and_var(vars_SND[0],
                                                                             vars_SND[1])
         return self._broadcast(f,[Fmu, Fvar], [])

    def predict_logdensity(self, Fmu, Fvar, Y):
        f = lambda vars_SND, vars_ND: self.likelihood.predict_logdensity(vars_SND[0],
                                                                       vars_SND[1],
                                                                       vars_ND[0])
        return self._broadcast(f,[Fmu, Fvar], [Y])
