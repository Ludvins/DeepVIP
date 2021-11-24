# Based on GPflow's likelihood

import torch
import numpy as np


class Likelihood(torch.nn.Module):
    def __init__(self, dtype=torch.float64, device=None):
        """"""
        super().__init__()
        self.dtype = dtype
        self.device = device

    @property
    def metrics(self):
        raise NotImplementedError

    def update_metrics(self, y, mean_pred, std_pred):
        raise NotImplementedError

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
        """"""
        super().__init__(dtype, device)
        self.log_variance = torch.nn.Parameter(
            torch.tensor(log_variance, dtype=dtype, device=self.device)
        )

        self.rmse_metric = torch.nn.MSELoss()
        self.nll_metric = torch.mean

    @property
    def metrics(self):
        return [self.rmse_metric, self.nll_metric]

    def update_metrics(self, y, mean_pred, std_pred):

        if mean_pred.ndim == 3:
            predictions = mean_pred.mean(0)
        else:
            predictions = mean_pred
        self.rmse_val = self.rmse_metric(y, predictions).sqrt()

        S = mean_pred.shape[0]
        normal = torch.distributions.Normal(loc=mean_pred, scale=std_pred)
        logpdf = normal.log_prob(y)
        nll = torch.logsumexp(logpdf, 0) - np.log(S)
        nll = -nll.mean()

        self.nll_val = nll

    def logdensity(self, x, mu, var):
        return -0.5 * (np.log(2 * np.pi) + var.log() + (mu - x).square() / var)

    def logp(self, F, Y):
        return self.logdensity(Y, F, self.log_variance.exp())

    def predict_mean_and_var(self, Fmu, Fvar):
        return Fmu, Fvar + self.log_variance.exp()

    def predict_density(self, Fmu, Fvar, Y):
        return self.logdensity(Y, Fmu, Fvar + self.log_variance.exp())

    def variational_expectations(self, Fmu, Fvar, Y):
        return (
            -0.5 * np.log(2 * np.pi)
            - 0.5 * self.log_variance
            - 0.5 * ((Y - Fmu).square() + Fvar) / self.log_variance.exp()
        )
