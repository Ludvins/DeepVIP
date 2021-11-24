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
        warmup_iterations=1,
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
        y_mean : float or array-like
                 Original value of the normalized labels
        y_std : float or array-like
                Original standar deviation of the normalized labels
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

        self.warmup_iterations = warmup_iterations

        self.device = device

        self.dtype = dtype

    def train_step(self, optimizer, X, y):
        """
        Defines the training step for the DVIP model.

        Parameters
        ----------
        data : tuple of shape
               ([num_samples, data_dim], [num_samples, labels_dim])
               Contains features and labels of the training set.

               Input labels must be standardized.

        Returns
        -------
        metrics : dictionary
                  Contains the resulting metrics of the training step.
        """

        if y.ndim == 1:
            y = y.unsqueeze(-1)

        if self.dtype != X.dtype:
            X = X.to(self.dtype)
        if self.dtype != y.dtype:
            y = y.to(self.dtype)

        optimizer.zero_grad()
        loss = self.nelbo(X, y)
        loss.backward()  # (retain_graph=True)

        optimizer.step()

        with torch.no_grad():
            means, variances = self.forward(X)

        self.likelihood.update_metrics(
            y * self.y_std + self.y_mean, means, variances
        )

        return loss

    def test_step(self, X, y):
        """
        Defines the test step for the DVIP model.

        Parameters
        ----------
        data : tuple of shape
               ([num_samples, data_dim], [num_samples, labels_dim])
               Contains features and labels of the training set.

        Returns
        -------
        metrics : dictionary
                  Contains the resulting metrics of the training step.
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
        predict_at : tf.tensor of shape (num_data, data_dim)
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
        X : tf.tensor of shape (num_data, data_dim)
            Contains the input features.
        full_cov : boolean
                   Whether to use the full covariance matrix or just
                   the diagonal values.

        Returns
        -------
        Fs : tf.tensor of shape (num_layers, num_data, data_dim)
             Contains the propagation made in each layer
        Fmeans : tf.tensor of shape (num_layers, num_data, output_dim)
                 Contains the mean value of the predictions at
                 each layer.
        Fvars : tf.tensor of shape (num_layers, num_data, output_dim)
                Contains the standard deviation of the predictions at
                each layer.
        Fprior : tf.tensor of shape (num_layers, S, num_data, num_outputs)
                 Contains the S prior samples for each layer at each data
                 point.
        """
        sX = torch.tile(X.unsqueeze(0), [num_samples, 1, 1])
        Fs, Fmeans, Fvars, Fpriors = [], [], [], []
        F = sX
        for layer in self.vip_layers:
            F, Fmean, Fvar, Fprior = layer.sample_from_conditional(
                F, full_cov=full_cov
            )

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
        predict_at : tf.tensor of shape (num_data, data_dim)
                     Contains the input features.
        full_cov : boolean
                   Whether to use the full covariance matrix or just
                   the diagonal values.
        Returns
        -------
        Fmeans : tf.tensor of shape (num_data, output_dim)
                 Contains the mean value of the predictions at
                 the last layer.
        Fvars : tf.tensor of shape (num_data, output_dim)
                Contains the standard deviation of the predictions at
                the last layer.

        """

        _, Fmeans, Fvars, _ = self.propagate(
            predict_at,
            num_samples,
            full_cov=full_cov,
        )
        return Fmeans[-1], Fvars[-1]

    def predict_y(self, predict_at, num_samples):
        Fmean, Fvar = self.predict_f(
            predict_at, num_samples=num_samples, full_cov=False
        )
        return self.likelihood.predict_mean_and_var(Fmean, Fvar)

    def get_prior_samples(self, X, full_cov=False):
        _, _, _, Fpriors = self.propagate(X, num_samples=1, full_cov=full_cov)

        Fpriors = torch.stack(Fpriors).squeeze(2)
        return Fpriors * self.y_std + self.y_mean

    def predict_log_density(self, data, num_samples):
        Fmean, Fvar = self.predict_f(
            data[0], num_samples=num_samples, full_cov=False
        )
        l = self.likelihood.predict_density(Fmean, Fvar, data[1])
        log_num_samples = torch.log(torch.Tensor(self.num_samples, self.dtype))

        return torch.logsumexp(l - log_num_samples, dim=0)

    def expected_data_log_likelihood(self, X, Y):
        """
        Compute expectations of the data log likelihood under the variational
        distribution with MC samples
        """
        F_mean, F_var = self.predict_f(
            X, num_samples=self.num_samples, full_cov=False
        )
        var_exp = self.likelihood.variational_expectations(
            F_mean, F_var, Y
        )  # Shape [S, N, D]
        return torch.mean(var_exp, dim=0)  # Shape [N, D]

    def nelbo(self, inputs, outputs):
        """
        Computes the evidence lower bound according to eq. (17) in the paper.
        :param data: Tuple of two tensors for input data X and labels Y.
        :return: Tensor representing ELBO.
        """
        X, Y = inputs, outputs

        likelihood = torch.sum(self.expected_data_log_likelihood(X, Y))

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
                    sections = name[: i + 1]

            padding = "\t" * (len(name) - 1)
            print(
                padding,
                "{}: {}".format(
                    name[-1], param.data.detach().cpu().numpy().flatten()
                ),
            )

        print("\n---------------------------\n\n")
