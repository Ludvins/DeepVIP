import torch
from .utils import reparameterize
import numpy as np
from .flows import *


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
        seed=2147483647,
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

        self.generator = torch.Generator(device)
        self.generator.manual_seed(seed)

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
            Targets of the given input.
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

    def forward(self, predict_at):
        """
        Computes the predicted labels for the given input.

        Parameters
        ----------
        predict_at : torch tensor of shape (batch_size, data_dim)
                     Contains the input features.

        Returns
        -------
        The predicted mean and standard deviation.
        """
        # Cast types if needed.
        if self.dtype != predict_at.dtype:
            predict_at = predict_at.to(self.dtype)

        mean, var = self.predict_y(predict_at, self.num_samples)
        # Return predictions scaled to the original scale.
        return mean * self.y_std + self.y_mean, torch.sqrt(var) * self.y_std

    def propagate(self, X, num_samples=1, return_prior_samples=False):
        """
        Propagates the input trough the layer.
        Propagates a sample of the predictive distribution of each layer
        through the next layer, iteratively.

        Parameters
        ----------
        X : torch tensor of shape (batch_size, data_dim)
            Contains the input features.
        num_samples : int
                      Number of MonteCarlo resamples to use
        return_prior_samples : bool
                               Wether to return the learned prior samples or not.

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
        sX = X.unsqueeze(0).transpose(0, -1)
        sX = torch.tile(sX, [num_samples]).transpose(0, -1)
        # Initialize arrays
        Fs, Fmeans, Fvars, Fpriors = [], [], [], []

        # The first input values are the original ones
        F = sX
        for layer in self.vip_layers:
            # Get input shape, S = MC resamples, N = num data, D 0 data dim
            S, N = F.shape[:2]
            D = F.shape[2:]
            # Flatten the MC resamples
            F_flat = torch.reshape(F, [S * N, *D])

            # Get the layer's predictive distribution and
            # prior samples if required.
            results = layer(F_flat, return_prior_samples)
            # Reshape predictions to the original shape
            D_out = results[0].shape[1:]
            Fmean = torch.reshape(results[0], [S, N, *D_out])
            Fvar = torch.reshape(results[1], [S, N, *D_out])

            # Use Gaussian re-parameterization trick to create samples.
            z = torch.randn(
                Fmean.shape,
                generator=self.generator,
                dtype=self.dtype,
                device=self.device,
            )
            F = reparameterize(Fmean, Fvar, z)

            # Store this layer results
            Fs.append(F)
            Fmeans.append(Fmean)
            Fvars.append(Fvar)

            # If prior samples are retrieved, store them in the original
            # shape.
            if return_prior_samples:
                Fprior = torch.reshape(
                    results[2],
                    [results[2].shape[0], S, N, *D_out],
                )
                Fpriors.append(Fprior)

        if return_prior_samples:
            return Fs, Fmeans, Fvars, Fpriors
        return Fs, Fmeans, Fvars

    def predict_f(self, predict_at, num_samples):
        """
        Returns the predicted mean and variance at the last layer.

        Parameters
        ----------
        predict_at : torch tensor of shape (batch_size, data_dim)
                     Contains the input features.
        num_samples : int
                      Number of MonteCarlo resamples to propagate.
        Returns
        -------
        Fmeans : torch tensor of shape (num_layers, batch_size, output_dim)
                 Contains the mean value of the predictions at
                 each layer.
        Fvars : torch tensor of shape (num_layers, batch_size, output_dim)
                Contains the standard deviation of the predictions at
                each layer.
        """

        _, Fmeans, Fvars = self.propagate(
            predict_at,
            num_samples,
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
        Fmean, Fvar = self.predict_f(predict_at, num_samples=num_samples)
        return self.likelihood.predict_mean_and_var(Fmean, Fvar)

    def predict_logdensity(self, X, Y):
        """
        Computes the model log likelihood on the given inputs and targets.

        Parameters
        ----------
        X : torch tensor of shape (N, data_dim)
            Contains the new input features.
        Y : torch tensor of shape (N, output_dim)
            Contains the new input targets.

        Returns
        -------
        The lof likelihood of the predictions for the given input values.
        """

        # Cast types if needed.
        if self.dtype != X.dtype:
            X = X.to(self.dtype)

        Fmean, Fvar = self.predict_f(X, num_samples=self.num_samples)
        l = self.likelihood.predict_logdensity(
            Fmean * self.y_std + self.y_mean, Fvar * self.y_std ** 2, Y
        )
        l = torch.sum(l, -1)
        log_num_samples = torch.log(torch.tensor(self.num_samples))

        return torch.mean(torch.logsumexp(l, axis=0) - log_num_samples)

    def get_prior_samples(self, X):
        """
        Returns the prior samples of the given inputs using 1 MonteCarlo
        resample.

        Parameters
        ----------
        X : torch tensor of shape (batch_size, data_dim)
            Contains the input features.

        Returns
        -------
        Fprior : torch tensor of shape (num_layers, S, batch_size, num_outputs)
                 Contains the S prior samples for each layer at each data
                 point.
        """
        _, _, _, Fpriors = self.propagate(X, num_samples=1, return_prior_samples=True)

        # Squeeze the MonteCarlo dimension
        Fpriors = torch.stack(Fpriors)[:, :, 0, :, :]
        # Scale data
        return Fpriors * self.y_std + self.y_mean

    def bb_alpha_energy(self, X, Y, alpha=0.5):
        """
        Compute Black-Box alpha energy objective function for
        the given inputs, targets and value of alpha.

        If alpha is 0, the standard variational evidence lower
        bound is computed.

        Parameters
        ----------
        X : torch tensor of shape (batch_size, data_dim)
            Contains the input features.
        y : torch tensor of shape (batch_size, output_dim)
            Targets of the given input, must be standardized.
        alpha : float between 0 and 1
                Value of alpha in BB-alpha energy.

        Returns
        -------
        bb_alpha : torch tensor of shape (batch_size, output_dim)
                   Contains the BB-alpha energy per data point.

        """
        # Compute model predictions, shape [S, N, D]
        F_mean, F_var = self.predict_f(X, num_samples=self.num_samples)

        # Compute variational expectation using Black-box alpha energy.
        # Shape [N, D]
        return self.likelihood.variational_expectations(F_mean, F_var, Y, alpha=alpha)

    def nelbo(self, X, y):
        """
        Computes the objective minimization function. When alpha is 0
        this function equals the variational evidence lower bound.
        Otherwise, Black-Box alpha inference is used.

        Parameters
        ----------
        X : torch tensor of shape (batch_size, data_dim)
            Contains the input features.
        y : torch tensor of shape (batch_size, output_dim)
            Targets of the given input, must be standardized.

        Returns
        -------
        nelbo : float
                Objective minimization function.
        """
        # Compute loss
        bb_alpha = self.bb_alpha_energy(X, y, self.bb_alpha)

        # Aggregate on data dimension
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
        """Prints the model variables in a prettier format."""
        import numpy as np

        print("\n---- MODEL PARAMETERS ----")
        np.set_printoptions(threshold=3, edgeitems=2)
        sections = []
        pad = "  "
        for name, param in self.named_parameters():
            name = name.split(".")
            for i in range(len(name) - 1):

                if name[i] not in sections:
                    print(pad * i, name[i].upper())
                    sections = name[: i + 1]

            padding = pad * (len(name) - 1)
            print(
                padding,
                "{}: ({})".format(name[-1], str(list(param.data.size()))[1:-1]),
            )
            print(
                padding + " " * (len(name[-1]) + 2),
                param.data.detach().cpu().numpy().flatten(),
            )

        print("\n---------------------------\n\n")


class TVIP(DVIP_Base):
    def __init__(
        self,
        likelihood,
        layer,
        num_data,
        num_samples,
        bb_alpha=0,
        y_mean=0.0,
        y_std=1.0,
        device=None,
        dtype=torch.float64,
        seed=2147483647,
    ):
        super().__init__(
            likelihood,
            [layer],
            num_data,
            num_samples=num_samples,
            bb_alpha=bb_alpha,
            y_mean=y_mean,
            y_std=y_std,
            device=device,
            dtype=dtype,
            seed=seed,
        )
        self.layer = layer
        self.KLs = []
        self.bb_alphas = []

    def propagate(self, X, num_samples=1, return_prior_samples=False):
        F, prior = self.layer(X, num_samples)

        if return_prior_samples:
            return F, torch.zeros_like(F), prior

        return F, torch.zeros_like(F)


    def predict_f(self, predict_at, num_samples):
        """
        Returns the predicted mean and variance at the last layer.

        Parameters
        ----------
        predict_at : torch tensor of shape (batch_size, data_dim)
                     Contains the input features.
        num_samples : int
                      Number of MonteCarlo resamples to propagate.
        Returns
        -------
        Fmeans : torch tensor of shape (num_layers, batch_size, output_dim)
                 Contains the mean value of the predictions at
                 each layer.
        Fvars : torch tensor of shape (num_layers, batch_size, output_dim)
                Contains the standard deviation of the predictions at
                each layer.
        """

        Fmeans, Fvars = self.propagate(
            predict_at,
            num_samples,
        )
        return Fmeans, Fvars

    def forward(self, predict_at, num_samples=None):
        """
        Computes the predicted labels for the given input.

        Parameters
        ----------
        predict_at : torch tensor of shape (batch_size, data_dim)
                     Contains the input features.

        Returns
        -------
        The predicted mean and standard deviation.
        """
        # Cast types if needed.
        if self.dtype != predict_at.dtype:
            predict_at = predict_at.to(self.dtype)

        if num_samples is None:
            num_samples = self.num_samples

        mean, var = self.predict_y(predict_at, num_samples)
        # Return predictions scaled to the original scale.
        return mean * self.y_std + self.y_mean, torch.sqrt(var) * self.y_std

    def get_prior_samples(self, X, num_samples):
        """
        Returns the prior samples of the given inputs using 1 MonteCarlo
        resample.

        Parameters
        ----------
        X : torch tensor of shape (batch_size, data_dim)
            Contains the input features.

        Returns
        -------
        Fprior : torch tensor of shape (num_layers, S, batch_size, num_outputs)
                 Contains the S prior samples for each layer at each data
                 point.
        """

        Fpriors, f = self.layer.forward_prior(X, num_samples)
        # Scale data
        return Fpriors * self.y_std + self.y_mean, f * self.y_std + self.y_mean

    def nelbo(self, X, y):
        """
        Computes the objective minimization function. When alpha is 0
        this function equals the variational evidence lower bound.
        Otherwise, Black-Box alpha inference is used.

        Parameters
        ----------
        X : torch tensor of shape (batch_size, data_dim)
            Contains the input features.
        y : torch tensor of shape (batch_size, output_dim)
            Targets of the given input, must be standardized.

        Returns
        -------
        nelbo : float
                Objective minimization function.
        """
        # Compute model predictions, shape [20, N, D_out]
        F, _ = self.layer(X, self.num_samples)
        # Compute variational expectation using Black-box alpha energy.
        # Shape [20, N]
        logpdf = self.likelihood.logp(F, y)
        if self.bb_alpha == 0:
            ve = torch.mean(logpdf, axis=0)
        if self.bb_alpha != 0:
            ve = (
                torch.logsumexp(self.bb_alpha * logpdf, axis=0)
                - torch.log(torch.tensor(F.shape[0]))
            ) / self.bb_alpha
        # Aggregate on data dimension
        ve = torch.sum(ve)

        # Scale loss term corresponding to minibatch size
        scale = self.num_data
        scale /= X.shape[0]

        # Compute KL term
        KL = self.layer.KL()
        self.bb_alphas.append((-scale * ve).detach().numpy())
        self.KLs.append(KL.detach().numpy())
        # print(-scale * ve)
        # print(KL)
        return -scale * ve + KL

    def freeze_prior(self):
        """Sets the prior parameters of each layer as non-trainable."""
        self.layer.freeze_prior()

    def freeze_posterior(self):
        """Sets the posterior parameters of every layer as non-trainable."""
        self.layer.freeze_posterior()

    def defreeze_prior(self):
        """Sets the prior parameters of each layer as non-trainable."""
        for layer in self.vip_layers:
            layer.defreeze_prior()

    def defreeze_posterior(self):
        """Sets the posterior parameters of every layer as non-trainable."""
        for layer in self.vip_layers:
            layer.defreeze_posterior()



class TDVIP(DVIP_Base):
    def __init__(
        self,
        likelihood,
        layers,
        num_data,
        num_samples,
        bb_alpha=0,
        y_mean=0.0,
        y_std=1.0,
        device=None,
        dtype=torch.float64,
        seed=2147483647,
    ):
        super().__init__(
            likelihood,
            layers,
            num_data,
            num_samples=num_samples,
            bb_alpha=bb_alpha,
            y_mean=y_mean,
            y_std=y_std,
            device=device,
            dtype=dtype,
            seed=seed,
        )
        self.KLs = []
        self.bb_alphas = []

    def propagate(self, X, num_samples=1):
        
        F, _ = self.vip_layers[0](X, num_samples)        
        F = F.reshape(-1, F.shape[-1])
        
        for layer in self.vip_layers[1:]:
            F, _ = layer(F, 1)
            F = F.squeeze(0)
        
        F = F.reshape(num_samples, -1, F.shape[-1])
        return F, torch.zeros_like(F)

    def propagate_prior(self, X, num_samples=1):
        
        F, f = self.vip_layers[0].forward_prior(X, num_samples)        
        F = F.reshape(-1, F.shape[-1])
        
        for layer in self.vip_layers[1:]:
            F, f = layer.forward_prior(F, 1)
            F = F.squeeze(0)
        
        F = F.reshape(num_samples, -1, F.shape[-1])
        return F, f

    def predict_f(self, predict_at, num_samples):
        """
        Returns the predicted mean and variance at the last layer.

        Parameters
        ----------
        predict_at : torch tensor of shape (batch_size, data_dim)
                     Contains the input features.
        num_samples : int
                      Number of MonteCarlo resamples to propagate.
        Returns
        -------
        Fmeans : torch tensor of shape (num_layers, batch_size, output_dim)
                 Contains the mean value of the predictions at
                 each layer.
        Fvars : torch tensor of shape (num_layers, batch_size, output_dim)
                Contains the standard deviation of the predictions at
                each layer.
        """

        Fmeans, Fvars = self.propagate(
            predict_at,
            num_samples,
        )
        return Fmeans, Fvars

    def forward(self, predict_at, num_samples=None):
        """
        Computes the predicted labels for the given input.

        Parameters
        ----------
        predict_at : torch tensor of shape (batch_size, data_dim)
                     Contains the input features.

        Returns
        -------
        The predicted mean and standard deviation.
        """
        # Cast types if needed.
        if self.dtype != predict_at.dtype:
            predict_at = predict_at.to(self.dtype)

        if num_samples is None:
            num_samples = self.num_samples

        mean, var = self.predict_y(predict_at, num_samples)
        # Return predictions scaled to the original scale.
        return mean * self.y_std + self.y_mean, torch.sqrt(var) * self.y_std

    def get_prior_samples(self, X, num_samples):
        """
        Returns the prior samples of the given inputs using 1 MonteCarlo
        resample.

        Parameters
        ----------
        X : torch tensor of shape (batch_size, data_dim)
            Contains the input features.

        Returns
        -------
        Fprior : torch tensor of shape (num_layers, S, batch_size, num_outputs)
                 Contains the S prior samples for each layer at each data
                 point.
        """

        Fpriors, f = self.propagate_prior(X, num_samples)
        # Scale data
        return Fpriors * self.y_std + self.y_mean, f * self.y_std + self.y_mean

    def nelbo(self, X, y):
        """
        Computes the objective minimization function. When alpha is 0
        this function equals the variational evidence lower bound.
        Otherwise, Black-Box alpha inference is used.

        Parameters
        ----------
        X : torch tensor of shape (batch_size, data_dim)
            Contains the input features.
        y : torch tensor of shape (batch_size, output_dim)
            Targets of the given input, must be standardized.

        Returns
        -------
        nelbo : float
                Objective minimization function.
        """
        # Compute model predictions, shape [20, N, D_out]
        F, _ = self.propagate(X, self.num_samples)
        # Compute variational expectation using Black-box alpha energy.
        # Shape [20, N]
        logpdf = self.likelihood.logp(F, y)
        if self.bb_alpha == 0:
            ve = torch.mean(logpdf, axis=0)
        if self.bb_alpha != 0:
            ve = (
                torch.logsumexp(self.bb_alpha * logpdf, axis=0)
                - torch.log(torch.tensor(F.shape[0]))
            ) / self.bb_alpha
        # Aggregate on data dimension
        ve = torch.sum(ve)

        # Scale loss term corresponding to minibatch size
        scale = self.num_data
        scale /= X.shape[0]

        # Compute KL term
        KL = 0
        for layer in self.vip_layers:
            KL += layer.KL()

        self.bb_alphas.append((-scale * ve).detach().numpy())
        self.KLs.append(KL.detach().numpy())
        # print(-scale * ve)
        # print(KL)
        return -scale * ve + KL

    def freeze_prior(self):
        """Sets the prior parameters of each layer as non-trainable."""
        self.layer.freeze_prior()

    def freeze_posterior(self):
        """Sets the posterior parameters of every layer as non-trainable."""
        self.layer.freeze_posterior()

    def defreeze_prior(self):
        """Sets the prior parameters of each layer as non-trainable."""
        for layer in self.vip_layers:
            layer.defreeze_prior()

    def defreeze_posterior(self):
        """Sets the posterior parameters of every layer as non-trainable."""
        for layer in self.vip_layers:
            layer.defreeze_posterior()

