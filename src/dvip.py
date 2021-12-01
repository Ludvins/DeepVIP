import torch

from src.layers import VIPLayer


class DVIP_Base(torch.nn.Module):
    def name(self):
        return "Deep VIP Base"

    def __init__(
        self,
        likelihood,
        layers,
        num_data,
        num_samples=1,
        y_mean=0.0,
        y_std=1.0,
        device=None,
        dtype=torch.float64,
    ):
        """
        Defines a Base class for Deep Variational Implicit Processes as a
        particular Keras model.

        Parameters
        ----------
        likelihood : Likelihood
                     Indicates the likelihood distribution of the data
        layers : array of Layer
                 Contains the different Variational Implicit Process layers
                 that make up this model.
        num_data : int
                   Amount of data samples
        num_samples : int
                      Number of Monte Carlo samples to broadcast by default.
        y_mean : float or array-like
                 Original value of the normalized labels
        y_std : float or array-like
                Original standar deviation of the normalized labels
        device : torch.device
                 The device in which the computations are made.
        dtype : data-type
                The dtype of the layer's computations and weights.
        """
        super().__init__()
        self.num_data = num_data
        self.y_mean = y_mean
        self.y_std = y_std

        self.num_samples = num_samples

        self.likelihood = likelihood
        self.vip_layers = torch.nn.ModuleList(layers)

        self.device = device
        self.dtype = dtype

    def train_step(self, optimizer, X, y):
        """
        Defines the training step for the DVIP model.

        Parameters
        ----------
        optimizer : torch.optim
                    The considered optimization algorithm.
        X : torch tensor of shape (batch_size, data_dim)
            Contains the input features.
        y : torch tensor of shape (batch_size, output_dim)
            Targets of the given input, must be standardized.

        Returns
        -------
        loss : float
               The nelbo of the model at the current state for the given inputs
        """

        # If targets are unidimensional,
        # ensure there is a second dimension (N, 1)
        if y.ndim == 1:
            y = y.unsqueeze(-1)

        # Transform inputs and largets to the model'd dtype
        if self.dtype != X.dtype:
            X = X.to(self.dtype)
        if self.dtype != y.dtype:
            y = y.to(self.dtype)

        # Clear gradients
        optimizer.zero_grad()

        # Compute loss
        loss = self.nelbo(X, y)
        # Create backpropagation graph
        loss.backward()
        # Make optimization step
        optimizer.step()

        # Create predictions to update the model's metrics. Turn off gradients
        # computations as they are not necessary.
        with torch.no_grad():
            means, variances = self.forward(X)
            # Update likelihood metrics using re-escaled targets.
            self.likelihood.update_metrics(y * self.y_std + self.y_mean, means,
                                           variances)

        return loss

    def test_step(self, X, y):
        """
        Defines the test step for the DVIP model.

        Parameters
        ----------
        X : torch tensor of shape (batch_size, data_dim)
            Contains the input features.
        y : torch tensor of shape (batch_size, output_dim)
            Targets of the given input, must be standardized.

        Returns
        -------
        loss : float
               The nelbo of the model at the current state for the given inputs
        """

        if y.ndim == 1:
            y = y.unsqueeze(-1)

        if self.dtype != X.dtype:
            X = X.to(self.dtype)
        if self.dtype != y.dtype:
            y = y.to(self.dtype)

        # Compute predictions
        with torch.no_grad():
            mean_pred, var_pred = self(X)  # Forward pass

            # Compute the loss
            loss = self.nelbo(X, (y - self.y_mean) / self.y_std)
            self.likelihood.update_metrics(y, mean_pred, var_pred)

        return loss

    def forward(self, predict_at):
        """
        Computes the predicted labels for the given input.

        Parameters
        ----------
        predict_at : torch tensor of shape (batch_size, data_dim)
                     Contains the input features.
        full_cov : boolean
                   Whether to use the full covariance matrix or just
                   the diagonal values.
        Returns
        -------
        The predicted labels using the model's likelihood
        and the predicted mean and standard deviation.
        """

        mean, var = self.predict_y(predict_at, self.num_samples)
        return mean * self.y_std + self.y_mean, var.sqrt() * self.y_std

    def propagate(self, X, num_samples=1, full_cov=False):
        """
        Propagates the input trough the layer, using the output of the previous
        as the input for the next one.

        Parameters
        ----------
        X : torch tensor of shape (batch_size, data_dim)
            Contains the input features.
        num_samples : int
                      Number of MonteCarlo resamples to use
        full_cov : boolean
                   Whether to use the full covariance matrix or just
                   the diagonal values.

        Returns
        -------
        Fs : torch tensor of shape (num_layers, batch_size, data_dim)
             Contains the propagation made in each layer
        Fmeans : torch tensor of shape (num_layers, batch_size, output_dim)
                 Contains the mean value of the predictions at
                 each layer.
        Fvars : torch tensor of shape (num_layers, batch_size, output_dim)
                Contains the standard deviation of the predictions at
                each layer.
        Fprior : torch tensor of shape (num_layers, S, batch_size, num_outputs)
                 Contains the S prior samples for each layer at each data
                 point.
        """
        # Replicate X in a new axis for MonteCarlo samples
        sX = torch.tile(X.unsqueeze(0), [num_samples, 1, 1])

        # Initialize arrays
        Fs, Fmeans, Fvars, Fpriors = [], [], [], []

        # The first input values are the original ones
        F = sX

        for layer in self.vip_layers:
            F, Fmean, Fvar, Fprior = layer.sample_from_conditional(
                F, full_cov=full_cov)

            Fs.append(F)
            Fmeans.append(Fmean)
            Fvars.append(Fvar)
            Fpriors.append(Fprior)

        return Fs, Fmeans, Fvars, Fpriors

    def predict_f(self, predict_at, num_samples, full_cov=False):
        """
        Returns the predicted mean and variance at the last layer.

        Parameters
        ----------
        predict_at : torch tensor of shape (batch_size, data_dim)
                     Contains the input features.
        num_samples : int
                      Number of MonteCarlo resamples to use
        full_cov : boolean
                   Whether to use the full covariance matrix or just
                   the diagonal values.
        Returns
        -------
        Fmeans : torch tensor of shape (num_layers, batch_size, output_dim)
                 Contains the mean value of the predictions at
                 each layer.
        Fvars : torch tensor of shape (num_layers, batch_size, output_dim)
                Contains the standard deviation of the predictions at
                each layer.
        """

        _, Fmeans, Fvars, _ = self.propagate(
            predict_at,
            num_samples,
            full_cov=full_cov,
        )
        return Fmeans[-1], Fvars[-1]

    def predict_y(self, predict_at, num_samples):
        """
        Returns the predicted mean and variance for the given inputs.
        Takes the predictions from the last layer and considers 
        applies the likelihood.

        Parameters
        ----------
        predict_at : torch tensor of shape (batch_size, data_dim)
                     Contains the input features.
        num_samples : int
                      Number of MonteCarlo resamples to use
        Returns
        -------
        Fmeans : torch tensor of shape (num_layers, batch_size, output_dim)
                 Contains the mean value of the predictions.
        Fvars : torch tensor of shape (num_layers, batch_size, output_dim)
                Contains the standard deviation of the predictions.
        """
        Fmean, Fvar = self.predict_f(predict_at,
                                     num_samples=num_samples,
                                     full_cov=False)
        return self.likelihood.predict_mean_and_var(Fmean, Fvar)

    def get_prior_samples(self, X, full_cov=False):
        """
        Returns the prior samples of the given inputs using 1 MonteCarlo 
        resample.

        Parameters
        ----------
        X : torch tensor of shape (batch_size, data_dim)
            Contains the input features.
        full_cov : boolean
                   Whether to use the full covariance matrix or just
                   the diagonal values.

        Returns
        -------
        Fprior : torch tensor of shape (num_layers, S, batch_size, num_outputs)
                 Contains the S prior samples for each layer at each data
                 point.
        """
        _, _, _, Fpriors = self.propagate(X, num_samples=1, full_cov=full_cov)

        # Squeeze the MonteCarlo dimension
        Fpriors = torch.stack(Fpriors).squeeze(2)
        # Scale data
        return Fpriors * self.y_std + self.y_mean

    def expected_data_log_likelihood(self, X, Y):
        """
        Compute expectations of the data log likelihood under the variational
        distribution with MC samples.
        
        Parameters
        ----------
        X : torch tensor of shape (batch_size, data_dim)
            Contains the input features.
        y : torch tensor of shape (batch_size, output_dim)
            Targets of the given input, must be standardized.

        """
        F_mean, F_var = self.predict_f(X,
                                       num_samples=self.num_samples,
                                       full_cov=False)
        var_exp = self.likelihood.variational_expectations(
            F_mean, F_var, Y)  # Shape [S, N, D]
        return torch.mean(var_exp, dim=0)  # Shape [N, D]

    def nelbo(self, X, y):
        """
        Computes the evidence lower bound.
        
        Parameters
        ----------
        X : torch tensor of shape (batch_size, data_dim)
            Contains the input features.
        y : torch tensor of shape (batch_size, output_dim)
            Targets of the given input, must be standardized.

        """
        likelihood = torch.sum(self.expected_data_log_likelihood(X, y))

        # scale loss term corresponding to minibatch size
        scale = self.num_data
        scale /= X.shape[0]

        # Compute KL term
        KL = torch.stack([layer.KL() for layer in self.vip_layers]).sum()

        return -(scale * likelihood - KL)

    @property
    def metrics(self):
        # We list our `Metric` objects here so that `reset_states()` can be
        # called automatically at the start of each epoch
        # or at the start of `evaluate()`.
        # If you don't implement this property, you have to call
        # `reset_states()` yourself at the time of your choosing.
        metrics = [self.loss_tracker]
        for metric in self.likelihood.metrics:
            metrics.append(metric)

        return metrics

    def print_variables(self):
        import numpy as np

        print("\n---- MODEL PARAMETERS ----")
        np.set_printoptions(threshold=3, edgeitems=2)
        sections = []
        for name, param in self.named_parameters():
            name = name.split(".")
            for i in range(len(name) - 1):

                if name[i] not in sections:
                    print("\t" * i, name[i].upper())
                    sections = name[:i + 1]

            padding = "\t" * (len(name) - 1)
            print(
                padding,
                "{}: {}".format(name[-1],
                                param.data.detach().cpu().numpy().flatten()),
            )

        print("\n---------------------------\n\n")
