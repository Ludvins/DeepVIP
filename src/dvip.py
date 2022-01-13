import torch


class DVIP_Base(torch.nn.Module):
    def name(self):
        return "Deep VIP Base"

    def __init__(
        self,
        likelihood,
        layers,
        num_data,
        num_samples=1,
        bb_alpha=0,
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
                     Indicates the likelihood distribution of the data.
        layers : array of Layer
                 Contains the different Variational Implicit Process layers
                 that define this model.
        num_data : int
                   Amount of data samples in the full dataset. This is used
                   to scale the likelihood in the loss function to the size
                   of the minibatch.
        num_samples : int
                      The number of samples to generate from the
                      posterior distribution.
        bb_alpha : float
                   Alpha value used for BlackBox alpha energy learning.
                   When 0, the usual ELBO from variational inference is used.
                   When 1, Expectation Propagation is apprixmated via MonteCarlo.
        y_mean : float or array-like
                 The given target values at training must be normalized.
                 This variable indicates the original mean value so that
                 the computed metrics follow the original scale.
        y_std : float or array-like
                Original standar deviation of the normalized targets.
        device : torch.device
                 The device in which the computations are made.
        dtype : data-type
                The dtype of the layer's computations and weights.

        Raises
        ------
        Warning
             When using only one layer, all posterior samples coincide.
             If the number of posterior samples is greater than one,
             a message is shown informing of this.
        """
        super().__init__()
        # Store data information
        self.num_data = num_data
        # Store Black-Box alpha value
        self.bb_alpha = bb_alpha

        # Store targets mean and std.
        self.y_mean = torch.tensor(y_mean, device=device)
        self.y_std = torch.tensor(y_std, device=device)

        # Set the amount of MC samples in training and test
        self.num_samples = num_samples

        # Store likelihood and Variational Implicit layers
        self.likelihood = likelihood
        self.vip_layers = torch.nn.ModuleList(layers)

        # Warning about vip_layers and posterior samples
        if len(self.vip_layers) == 1 and self.num_samples > 1:
            import warnings

            self.num_samples = 1
            warnings.warn(
                "Using more than one posterior sample seriously affects the"
                " computational time. When wsing only one layer all posterior"
                " samples coincide. The number of samples will be set to 1."
            )

        # Set device and data type (precision)
        self.device = device
        self.dtype = dtype

    def train_step(self, optimizer, X, y):
        """
        Defines the training step for the DVIP model using a simple optimizer.
        This method illustrates a standard training step. If more complex
        operations are needed, such as optimizers with double steps,
        create your own training step, calling this one is not compulsory.

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
               The nelbo of the model at the current state for the
               given inputs.
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
        # torch.nn.utils.clip_grad_norm_(self.parameters(), 10)

        # Make optimization step
        optimizer.step()

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

        # In case targets are one-dimensional and flattened, add a final dimension.
        if y.ndim == 1:
            y = y.unsqueeze(-1)

        # Cast types if needed.
        if self.dtype != X.dtype:
            X = X.to(self.dtype)
        if self.dtype != y.dtype:
            y = y.to(self.dtype)

        # Compute predictions
        with torch.no_grad():
            mean_pred, std_pred = self(X)  # Forward pass

            # Temporarily change the num data variable so that the
            # scale of the likelihood is correctly computed on the
            # test dataset.
            num_data = self.num_data
            self.num_data = X.shape[0]
            # Compute the loss with scaled data
            loss = self.nelbo(X, (y - self.y_mean) / self.y_std)
            self.num_data = num_data

        return loss, mean_pred, std_pred

    def forward(self, predict_at, full_cov=False):
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

        mean, var = self.predict_y(predict_at, self.num_samples, full_cov=full_cov)
        # Return predictions scaled to the original scale.
        return mean * self.y_std + self.y_mean, torch.sqrt(var) * self.y_std

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
                   the diagonal values. Full covariances is not
                   supported by now.

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
            F, Fmean, Fvar, Fprior = layer.sample_from_conditional(F)

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
                      Number of MonteCarlo resamples to propagate.
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

    def predict_y(self, predict_at, num_samples, full_cov=False):
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
        full_cov : boolean
                   Whether to use the full covariance matrix or just
                   the diagonal values.


        Returns
        -------
        Fmeans : torch tensor of shape (num_layers, batch_size, output_dim)
                 Contains the mean value of the predictions.
        Fvars : torch tensor of shape (num_layers, batch_size, output_dim)
                Contains the standard deviation of the predictions.
        """
        Fmean, Fvar = self.predict_f(
            predict_at, num_samples=num_samples, full_cov=full_cov
        )

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
        _, _, _, Fpriors = self.propagate(
            X, num_samples=self.num_samples, full_cov=full_cov
        )

        # Squeeze the MonteCarlo dimension
        Fpriors = torch.stack(Fpriors)[:, :, 0, :, :]
        # Scale data
        return Fpriors * self.y_std + self.y_mean

    def expected_data_log_likelihood(self, X, Y):
        """
        Compute expectations of the expected data log likelihood
        under the variational distribution with MC samples.

        Parameters
        ----------
        X : torch tensor of shape (batch_size, data_dim)
            Contains the input features.
        y : torch tensor of shape (batch_size, output_dim)
            Targets of the given input, must be standardized.

        """
        # Compute model predictions, shape [S, N, D]
        F_mean, F_var = self.predict_f(X, num_samples=self.num_samples, full_cov=False)
        # Compute variational expectation. Shape [S, N, D]
        var_exp = self.likelihood.variational_expectations(F_mean, F_var, Y)
        # As the mixture in the predictive distribution gives
        # equal probability to each mixture, averaging is
        # perfirmed using a mean.
        return torch.mean(var_exp, dim=0)  # Shape [N, D]

    def bb_alpha_energy(self, X, Y, alpha=0.5):
        """
        Compute Black-Box alpha energy obejective function for
        the given inputs, targets and value of alpha.

        This value is estimated using MonteCarlo if different
        from zero.
        """
        # When alpha is zero, this coincides with the variational
        # expected log likelihood.
        if alpha == 0:
            return self.expected_data_log_likelihood(X, Y)

        # Compute predictive mixtures.
        F_mean, F_var = self.predict_f(X, num_samples=self.num_samples, full_cov=False)
        # Create MonteCarlo samples equally distributed between
        # the mixture components.
        n_mixtures = F_mean.shape[0]
        MC_samples = 100
        if MC_samples < n_mixtures:
            MC_samples = n_mixtures

        # Set random generator
        generator = torch.Generator(self.device)
        generator.manual_seed(2147483647)

        # Given that all distributions in the predictive mixture are equally
        # probable, generate the same number of samples of each one.
        z = torch.randn(
            [MC_samples // n_mixtures, *F_mean.shape],
            generator=generator,
            dtype=self.dtype,
            device=self.device,
        )
        samples = F_mean + z * torch.sqrt(F_var)
        # Flatten the mixture dimension. Shape [MC_samples, N, D_out]
        samples = samples.reshape([MC_samples, *samples.shape[-2:]])
        # Compute log pdf of the targets. Shape [MC_samples, N, D_out]
        log_pdf = self.likelihood.logp(samples, Y)

        # Compute alpha energy
        log_expected = torch.logsumexp(alpha * log_pdf, 0) - torch.tensor(
            MC_samples, dtype=self.dtype
        )

        return log_expected / alpha

    def nelbo(self, X, y):
        """
        Computes the objective minimization function. When alpha is 0
        this function equals the variational evidence lower bound.
        Othewise, Black-Box alpha inference is used, estimating the
        objective with MonteCarlo samples from the predictive mixture
        distribution.

        Parameters
        ----------
        X : torch tensor of shape (batch_size, data_dim)
            Contains the input features.
        y : torch tensor of shape (batch_size, output_dim)
            Targets of the given input, must be standardized.

        """
        # Compute loss
        bb_alpha = self.bb_alpha_energy(X, y, alpha=self.bb_alpha)
        # Agregate on data dimension
        bb_alpha = torch.sum(bb_alpha)

        # Scale loss term corresponding to minibatch size
        scale = self.num_data
        scale /= X.shape[0]
        # Compute KL term
        KL = torch.stack([layer.KL() for layer in self.vip_layers]).sum()
        return -scale * bb_alpha + KL

    def freeze_posterior(self):
        """Sets the posterior parameters of every layer as non-trainable."""
        for layer in self.vip_layers:
            layer.freeze_posterior()

    def freeze_prior(self):
        """Sets the prior parameters of each layer as non-trainable."""
        for layer in self.vip_layers:
            layer.freeze_prior()

    def freeze_ll_variance(self):
        """Sets the likelihood variance as a non-trainable parameter."""
        self.likelihood.log_variance.requires_grad = False

    def print_variables(self):
        """Prints the model variables in a prettier way."""
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
                "{}: {}".format(name[-1], param.data.detach().cpu().numpy().flatten()),
            )

        print("\n---------------------------\n\n")
