import torch


class DVIP_Base(torch.nn.Module):
    def name(self):
        return "Deep VIP Base"

    def __init__(self,
                 likelihood,
                 layers,
                 num_data,
                 num_samples=1,
                 y_mean=0.0,
                 y_std=1.0,
                 warmup_iterations=1,
                 device=None,
                 dtype=torch.float64,
                 **kwargs):
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
        **kwargs : dict, optional
                   Extra arguments to `Model`.
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

        # Metric trackers
        #self.loss_tracker = tf.keras.metrics.Mean(name="nelbo")

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

        loss = self.nelbo(X, y)

        optimizer.zero_grad()
        loss.backward()  #(retain_graph=True)
        optimizer.step()

        with torch.no_grad():
            means, variances = self.forward(X)

        self.likelihood.update_metrics(y * self.y_std + self.y_mean, means,
                                       variances)

        return loss

    def predict(self, inputs):

        output_means, output_vars = self(inputs)
        output_means = output_means.mean(0)
        output_vars = output_vars.mean(0)
        return output_means * self.y_std + self.y_mean, output_vars.sqrt(
        ) * self.y_std

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

            # Update the metrics
            #self.loss_tracker.update_state(loss)

        return loss

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

        # Define arrays
        Fs, Fmeans, Fvars, Fprior = [], [], [], []

        # First input corresponds to the original one
        F = sX

        for layer in self.vip_layers:
            F, Fmean, Fvar, prior_samples = layer.sample_from_conditional(
                F, full_cov=full_cov)

            # Store values
            Fs.append(F)
            Fmeans.append(Fmean)
            Fvars.append(Fvar)
            Fprior.append(prior_samples)

        # Return arrays
        return Fs, Fmeans, Fvars, Fprior

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

    def predict_prior_samples(self, predict_at, full_cov=False):
        """
        Returns the generated prior samples at the last layer.

        Parameters
        ----------
        predict_at : tf.tensor of shape (num_data, data_dim)
                     Contains the input features.
        full_cov : boolean
                   Whether to use the full covariance matrix or just
                   the diagonal values.
        Returns
        -------
        Fprior : tf.tensor of shape (S, num_data, output_dim)
                 Contains the generated S samples. This values must
                 be scaled.
        """

        _, _, _, Fprior = self.propagate(
            predict_at,
            num_samples=1,
            full_cov=full_cov,
        )
        Fprior = torch.permute(torch.cat(Fprior), (1, 0, 2, 3))
        return Fprior * self.y_std + self.y_mean

    def forward(self, predict_at, full_cov=False):
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

        mean, var = self.predict_f(predict_at,
                                   self.num_samples,
                                   full_cov=full_cov)
        return self.likelihood.predict_mean_and_var(mean, var)

    def predict_log_density(self, data, num_samples):
        Fmean, Fvar = self.predict_f(data[0], num_samples, full_cov=False)
        l = self.likelihood.predict_logdensity(Fmean, Fvar, data[1])
        log_num_samples = torch.log(torch.Tensor(self.num_samples, self.dtype))

        return torch.logsumexp(l - log_num_samples, dim=0)

    def expected_data_log_likelihood(self, X, Y):
        """
        Compute expectations of the data log likelihood under the variational
        distribution

        Parameters
        ----------
        X : tf.tensor of shape (num_samples, input_dim)
            Contains input features

        Y : tf.tensor of shape (num_samples, output_dim)
            Contains input labels

        Returns
        -------
        var_exp : tf.tensor of shape (1)
                  Contains the variational expectation

        """
        F_mean, F_var = self.predict_f(X,
                                       num_samples=self.num_samples,
                                       full_cov=False)
        var_exp = self.likelihood.variational_expectations(F_mean, F_var, Y)

        return torch.mean(var_exp, dim=0)  # Shape (N, D)

    def nelbo(self, inputs, outputs):
        """
        Computes the evidence lower bound.

        Parameters
        ----------
        inputs : tf.tensor of shape (num_samples, input_dim)
                 Contains input features

        outputs : tf.tensor of shape (num_samples, output_dim)
                  Contains input labels

        Returns
        -------
        nelbo : tf.tensor
                Negative evidence lower bound
        """
        X, Y = inputs, outputs

        likelihood = torch.mean(self.expected_data_log_likelihood(X, Y))

        # scale loss term corresponding to minibatch size
        scale = self.num_data / X.shape[0]
        # Compute KL term
        KL = torch.stack([layer.KL() for layer in self.vip_layers]).sum()

        return -scale * likelihood + KL

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