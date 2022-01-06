#!/usr/bin/env python3
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from scipy.stats import norm


class Metrics:
    def __init__(
        self,
    ):
        """Defines a class that encapsulates all considered metrics."""
        self.rmse = tf.keras.metrics.RootMeanSquaredError(name="rmse")
        self.nll = tf.keras.metrics.Mean(name="nll")
        self.crps = tf.keras.metrics.Mean(name="crps")
        self.loss = tf.keras.metrics.Mean(name="loss")
        self.reset()

    def reset(self):
        """ Ressets all the metrics to zero. """
        self.loss.reset_state()
        self.rmse.reset_state()
        self.nll.reset_state()
        self.crps.reset_state()

    def update(self, y, loss, mean_pred, std_pred, light=True):
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
        # Update light metrics
        self.loss.update_state(loss)
        # The RMSE is computed using the mean prediction of the Gaussian
        #  Mixture, that is, the mean of the mean predictions.
        self.rmse.update_state(y, tf.reduce_mean(mean_pred, 0))
        self.nll.update_state(self.compute_nll(y, mean_pred, std_pred))

        # Update heavy metrics
        if not light:
            self.crps.update_state(
                self.compute_crps(
                    y.numpy(), mean_pred.numpy(), std_pred.numpy()
                )
            )

    def compute_nll(self, y, mean_pred, std_pred):
        """Computes the negative log likelihood for the given predictions.
        Assumes Gaussian likelihood."""
        # Get the number of posterior samples
        S = mean_pred.shape[0]
        # Compute the Gaussian likelihood of the data in each sample
        normal = tfp.distributions.Normal(loc=mean_pred, scale=std_pred)
        logpdf = normal.log_prob(y)
        # Sum label dimensionality
        logpdf = tf.reduce_sum(logpdf, -1)
        # Compute the Negative log-likelihood on the Gaussian mixture
        ll = tf.math.reduce_logsumexp(logpdf, 0) - tf.math.log(
            tf.cast(S, tf.float64)
        )
        return -tf.reduce_mean(ll)

    def compute_crps(self, y, mean_pred, std_pred):

        if mean_pred.shape[-1] != 1:
            # Multidimensional output not implemented yet
            raise NotImplementedError

        mean_pred = np.squeeze(mean_pred, -1)
        std_pred = np.squeeze(std_pred, -1)

        # Define the auxiliary function to help with the calculations
        def A(mu, sigma_2):
            first_term = (
                2 * np.sqrt(sigma_2) * norm.pdf(mu / tf.math.sqrt(sigma_2))
            )
            sec_term = mu * (2 * norm.cdf(mu / tf.math.sqrt(sigma_2)) - 1)
            return first_term + sec_term

        # Estimate the differences between means and variances for each sample, batch-wise
        var_pred = std_pred ** 2
        n_mixtures = mean_pred.shape[0]
        batch_size = mean_pred.shape[1]
        crps_exact = 0.0

        for i in range(batch_size):
            means_vec = mean_pred[:, i]
            vars_vec = var_pred[:, i]

            means_diff = np.zeros(
                (n_mixtures, n_mixtures),
            )
            vars_sum = np.zeros(
                (n_mixtures, n_mixtures),
            )
            ru, cu = np.triu_indices(n_mixtures, 1)
            rl, cl = np.tril_indices(n_mixtures, 1)

            means_diff[ru, cu] = (
                means_vec[tf.constant(ru)] - means_vec[tf.constant(cu)]
            )
            means_diff[rl, cl] = means_vec[rl] - means_vec[cl]
            vars_sum[ru, cu] = vars_vec[ru] + vars_vec[cu]
            vars_sum[rl, cl] = vars_vec[rl] + vars_vec[cl]

            # Term only depending on the means and vars
            fixed_term = 1 / 2 * np.mean(A(means_diff, vars_sum))

            # Term that depends on the real value of the data
            dev_mean = y[i] - means_vec
            data_term = np.mean(A(dev_mean, vars_vec))

            crps_exact += data_term - fixed_term

        return crps_exact / batch_size

    def get_dict(self):
        return {
            "LOSS": self.loss.result().numpy(),
            "RMSE": self.rmse.result().numpy(),
            "NLL": self.nll.result().numpy(),
            "CRPS": self.crps.result().numpy(),
        }
