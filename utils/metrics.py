#!/usr/bin/env python3
import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from properscoring import crps_gaussian


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
        l = likelihood.logdensity(mean_pred, std_pred ** 2, y)
        l = torch.sum(l, -1)
        return -torch.mean(l)


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
        likelihood : instance of Likelihood
                     Usable to compute the log likelihood metric.
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
        self.mse += scale * self.compute_mse(y, mean_pred)
        self.nll += scale * self.compute_nll(y, mean_pred, std_pred, likelihood)
        # Update heavy metrics
        if not light:
            self.crps += scale * self.compute_crps(y, mean_pred, std_pred)


    def compute_mse(self, y, prediction):
        """Computes the root mean squared error for the given predictions."""
        return torch.nn.functional.mse_loss(prediction, y)

    def compute_crps(self, y, mean_pred, std_pred):
        crps= crps_gaussian(y.detach().cpu(), mean_pred.detach().cpu(), std_pred.detach().cpu())
        return np.mean(np.sum(crps, -1))

    def get_dict(self):
        return {
            "LOSS": float(self.loss.detach().cpu().numpy()),
            "RMSE": np.sqrt(self.mse.detach().cpu().numpy()),
            "NLL": float(self.nll.detach().cpu().numpy()),
            "CRPS": float(self.crps),
        }


class MetricsClassification(Metrics):
    def __init__(self, num_data=-1, device=None):
        super().__init__(num_data, device)

    def reset(self):
        """Ressets all the metrics to zero."""
        super().reset()
        self.acc = torch.tensor(0.0, device=self.device)
        self.auc = torch.tensor(0.0, device=self.device)

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
        likelihood : instance of Likelihood
                     Usable to compute the log likelihood metric.
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
        self.auc += scale * self.compute_auc(y, mean_pred)

    def compute_auc(self, y_true, prediction):
        if prediction.shape[-1] == 1:
            pred = torch.mean(prediction, 0)
        else:
            return 0
        return roc_auc_score(y_true, pred, multi_class="ovr")

    def compute_acc(self, y, prediction):
        """"""
        if prediction.shape[-1] == 1:
            pred = torch.mode(torch.round(prediction), 0)[0].flatten()
        else:
            pred = torch.mode(torch.argmax(prediction, -1), 0)[0]
        return (pred == y.flatten()).float().mean()

    def get_dict(self):
        return {
            "LOSS": float(self.loss.detach().cpu().numpy()),
            "NLL": float(self.nll.detach().cpu().numpy()),
            "Error": 1 - float(self.acc.detach().cpu().numpy()),
            "AUC": float(self.auc.detach().cpu().numpy()),
        }
