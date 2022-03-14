#!/usr/bin/env python3
import numpy as np
import torch

class Metrics:
    def __init__(self, num_data=-1, device=None):
        """Defines a class that encapsulates all considered metrics.

        Arguments
        ---------
        num_data : int
                   Number of data samples in the dataset used. This is used
                   to scale the metrics from each batch by the proportion of
                   data they contain.
                   This scale is never greater than 1.
        device : torch device
                 Device in which to make the computations.
        """

        self.num_data = num_data
        self.device = device
        self.reset()

    def reset(self):
        """Ressets all the metrics to zero."""
        self.loss = torch.tensor(0.0, device=self.device)
        self.nll = torch.tensor(0.0, device=self.device)

    def update(self, y, loss, mean_pred, std_pred, likelihood, light=True):
        raise NotImplementedError
    
    def compute_nll(self, y, mean_pred, std_pred, likelihood):
        l = likelihood.logdensity(y, mean_pred, std_pred**2)
        l = torch.sum(l, -1)
        log_num_samples = torch.log(torch.tensor(mean_pred.shape[0]))
        lse =  torch.logsumexp(l , axis=0)- log_num_samples
        return -torch.mean(lse)


class MetricsRegression(Metrics):
    def __init__(self, num_data=-1, device=None):
        super().__init__(num_data, device)

    def reset(self):
        """Ressets all the metrics to zero."""
        super().reset()
        self.mse = torch.tensor(0.0, device=self.device)
        self.crps = torch.tensor(0.0, device=self.device)

    def update(self, y, loss, mean_pred, std_pred, likelihood, light=True):
        """Updates all the metrics given the results in the parameters.

        Arguments
        ---------

        y : torch tensor of shape (batch_size, output_dim)
            Contains the true targets of the data.
        loss : torch tensor of shape ()
               Contains the loss value for the given batch.
        mean_pred : torch tensor of shape (S, batch_size, output_dim)
                    Contains the mean predictions for each sample
                    in the batch.
        std_pred : torch tensor of shape (S, batch_size, output_dim)
                   Contains the std predictions for each sample
                   in the batch.
        light : boolean
                Wether to compute only the lighter (computationally) metrics.
        """
        # Conmpute the scale value using the batch_size
        batch_size = y.shape[0]
        if self.num_data == -1:
            scale = 1
        else:
            scale = batch_size / self.num_data
        # Update light metrics
        self.loss += scale * loss
        # The RMSE is computed using the mean prediction of the Gaussian
        #  Mixture, that is, the mean of the mean predictions.
        self.mse += scale * self.compute_mse(y, mean_pred.mean(0))
        self.nll += scale * self.compute_nll(y, mean_pred, std_pred, likelihood)
        # Update heavy metrics
        if not light:
            self.crps += scale * self.compute_crps(y, mean_pred, std_pred)

    def compute_mse(self, y, prediction):
        """Computes the root mean squared error for the given predictions."""
        return torch.nn.functional.mse_loss(prediction, y)
    
    def compute_crps(self, y, mean_pred, std_pred):

        if mean_pred.shape[-1] != 1:
            # Multidimensional output not implemented yet
            raise NotImplementedError

        mean_pred = mean_pred.squeeze(-1)
        std_pred = std_pred.squeeze(-1)

        # Define the auxiliary function to help with the calculations
        def A(mu, sigma_2):
            norm = torch.distributions.normal.Normal(
                torch.zeros_like(mu), torch.ones_like(mu)
            )
            first_term = (
                2
                * torch.sqrt(sigma_2)
                * torch.exp(norm.log_prob(mu / torch.sqrt(sigma_2)))
            )
            sec_term = mu * (2 * norm.cdf(mu / torch.sqrt(sigma_2)) - 1)
            return first_term + sec_term

        # Estimate the differences between means and variances for each sample, batch-wise
        var_pred = std_pred ** 2
        n_mixtures = mean_pred.shape[0]
        batch_size = mean_pred.shape[1]
        crps_exact = 0.0

        for i in range(batch_size):
            means_vec = mean_pred[:, i]
            vars_vec = var_pred[:, i]

            means_diff = torch.zeros(
                (n_mixtures, n_mixtures),
                dtype=torch.float64,
                device=self.device,
            )
            vars_sum = torch.zeros(
                (n_mixtures, n_mixtures),
                dtype=torch.float64,
                device=self.device,
            )
            ru, cu = torch.triu_indices(n_mixtures, n_mixtures, 1)
            rl, cl = torch.tril_indices(n_mixtures, n_mixtures, 1)

            means_diff[ru, cu] = means_vec[ru] - means_vec[cu]
            means_diff[rl, cl] = means_vec[rl] - means_vec[cl]
            vars_sum[ru, cu] = vars_vec[ru] + vars_vec[cu]
            vars_sum[rl, cl] = vars_vec[rl] + vars_vec[cl]

            # Term only depending on the means and vars
            fixed_term = 1 / 2 * torch.mean(A(means_diff, vars_sum))

            # Term that depends on the real value of the data
            dev_mean = y[i] - means_vec
            data_term = torch.mean(A(dev_mean, vars_vec))

            crps_exact += data_term - fixed_term

        return crps_exact / batch_size

    def get_dict(self):
        return {
            "LOSS": float(self.loss.detach().cpu().numpy()),
            "RMSE": np.sqrt(self.mse.detach().cpu().numpy()),
            "NLL": float(self.nll.detach().cpu().numpy()),
            "CRPS": float(self.crps.detach().cpu().numpy()),
        }



class MetricsClassification(Metrics):
    def __init__(self, num_data=-1, device=None):
        super().__init__(num_data, device)

    def reset(self):
        """Ressets all the metrics to zero."""
        super().reset()
        self.acc = torch.tensor(0.0, device=self.device)

    def update(self, y, loss, mean_pred, std_pred, likelihood, light = True):
        """Updates all the metrics given the results in the parameters.

        Arguments
        ---------

        y : torch tensor of shape (batch_size, output_dim)
            Contains the true targets of the data.
        loss : torch tensor of shape ()
               Contains the loss value for the given batch.
        pred : torch tensor of shape (S, batch_size, output_dim)
               Contains the predictions for each sample
               in the batch.
        """
        # Conmpute the scale value using the batch_size
        batch_size = y.shape[0]
        if self.num_data == -1:
            scale = 1
        else:
            scale = batch_size / self.num_data
        # Update light metrics
        self.loss += scale * loss
        self.acc += scale * self.compute_acc(y, mean_pred)
        self.nll += scale * self.compute_nll(y, mean_pred, std_pred, likelihood)

    def compute_nll(self, y, mean_pred, std_pred, likelihood):
        l = likelihood.logdensity( mean_pred, std_pred**2, y)
        l = torch.sum(l, -1)
        log_num_samples = torch.log(torch.tensor(mean_pred.shape[0]))
        lse =  torch.logsumexp(l , axis=0)- log_num_samples
        return -torch.mean(lse)

    def compute_acc(self, y, prediction):
        """"""
        #mode(np.argmax(m, 2), 0)[0].reshape(Y_batch.shape).astype(int)==Y_batch.astype(int))
        pred = torch.mode(torch.argmax(prediction, -1), 0)[0]
        return (pred == y.flatten()).float().mean()

    def get_dict(self):
        return {
            "LOSS": float(self.loss.detach().cpu().numpy()),
            "NLL": float(self.nll.detach().cpu().numpy()),
            "Error": 1 - float(self.acc.detach().cpu().numpy()),
        }
