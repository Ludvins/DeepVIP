import torch
from .utils import reparameterize
import numpy as np
import torch.autograd.profiler as profiler


class BaseVaLLA(torch.nn.Module):
    """
    Defines a Sparse Linearized Laplace Approximation model.

    Parameters
    ----------
    net_forward : Callable
                  Forward method of the deep model on which LLA is being applied.
    Z : array of size (num_inducing, input_dim)
        Contains the inducing locations of the model.
    prior_std : float
                Value of the standrd deviation of the Gaussian prior over parameters.
    likelihood : Likelihood
                 Indicates the likelihood distribution of the data.
    num_data : int
                Amount of data samples in the full dataset. This is used
                to scale the likelihood in the loss function to the size
                of the minibatch.
    output_dim : int
                 Dimensionality of the targets.
    backend : Callable
              Returns the Jacobian of the deep model with respect to the given input.
    track_inducing_locations : Boolean
                               If True, an history of the inducing locations is stored.
    fix_inducing_locations : Boolean
                             If True, the inducing locations are fixed to their initial
                             value.
    alpha : float
            Alpha value used for BlackBox alpha energy learning.
            When 0, the usual ELBO from variational inference is used.
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
    """

    def __init__(
        self,
        net,
        Z,
        prior_std,
        num_data,
        output_dim,
        track_inducing_locations=False,
        inducing_classes=None,
        y_mean=0.0,
        y_std=1.0,
        alpha=0.5,
        device=None,
        dtype=torch.float64,
    ):
        super().__init__()
        # Store data information
        self.num_data = num_data
        self.output_dim = output_dim
        # Store targets mean and std.
        self.y_mean = torch.tensor(y_mean, device=device)
        self.y_std = torch.tensor(y_std, device=device)

        self.alpha = alpha
        self.net = net
        self.num_inducing = Z.shape[0]

        self.prior_std = torch.tensor(prior_std, device=device, dtype=dtype)
        self.prior_std = torch.nn.Parameter(self.prior_std)

        self.inducing_locations = torch.tensor(Z, device=device, dtype=dtype)
        self.inducing_locations = torch.nn.Parameter(self.inducing_locations)

        if inducing_classes is None:
            self.inducing_class = torch.tensor(
                np.tile(
                    np.arange(self.output_dim),
                    reps=np.ceil(Z.shape[0] / self.output_dim).astype(int),
                )[: Z.shape[0]],
                device=device,
                dtype=torch.long,
            )
        else:
            self.inducing_class = torch.tensor(
                inducing_classes, device=device, dtype=torch.long
            )

        self.track_inducing_locations = track_inducing_locations
        if track_inducing_locations:
            self.inducing_history = [
                self.inducing_locations.clone().detach().cpu().numpy()
            ]

        # Set device and data type (precision)
        self.device = device
        self.dtype = dtype

        # Initialize cholesky decomposition of identity
        I = np.eye(self.num_inducing)

        li, lj = torch.tril_indices(self.num_inducing, self.num_inducing)
        triangular_q_sqrt = I[li, lj]
        self.L = torch.tensor(
            triangular_q_sqrt,
            dtype=self.dtype,
            device=self.device,
        )

        self.L = torch.nn.Parameter(self.L)
        self.ell_history = []
        self.kl_history = []

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

        if self.track_inducing_locations:
            self.inducing_history += [
                self.inducing_locations.clone().detach().cpu().numpy()
            ]

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
               The nelbo of the model at the current state for the given inputs.
        mean_pred : torch tensor of size (batch_size, output_dim)
                    Predictive mean of the model on the given batch
        var_pred : torch tensor of size (batch_size, output_dim, output_dim)
                   Contains the covariance matrix of the model for each element on
                   the batch.
        """
        with torch.no_grad():
            # In case targets are one-dimensional and flattened, add a final dimension.
            if y.ndim == 1:
                y = y.unsqueeze(-1)

            # Cast types if needed.
            if self.dtype != X.dtype:
                X = X.to(self.dtype)
            if self.dtype != y.dtype:
                y = y.to(self.dtype)

            Fmean, Fvar = self(X)  # Forward pass

            # Temporarily change the num data variable so that the
            # scale of the likelihood is correctly computed on the
            # test dataset.
            num_data = self.num_data
            self.num_data = X.shape[0]
            # Compute the loss with scaled data
            loss = self.nelbo(X, (y - self.y_mean) / self.y_std)
            self.num_data = num_data

            return loss, Fmean, Fvar

    def compute_kernels(self, X):
        _, Kx_diag, Kxz, Kz = self.net.get_full_kernels(
            X,
            self.inducing_locations,
            self.inducing_class,
        )
        Kx_diag = self.prior_std**2 * Kx_diag
        Kxz = self.prior_std**2 * Kxz
        Kz = self.prior_std**2 * Kz
        F_mean = self.net(X)
        return F_mean, Kx_diag, Kxz, Kz

    def forward(self, X):
        raise NotImplementedError

    def compute_KL(self):
        """
        Computes the Kulback-Leibler divergence between the variational distribution
        and the prior.
        """

        log_det = torch.logdet(self.H)
        trace = torch.sum(torch.diagonal(self.Kz @ self.A))
        KL = 0.5 * log_det - 0.5 * trace
        return torch.sum(KL)

    def nelbo(self, X, y):
        raise NotImplementedError

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

    def freeze_cholesky(self):
        self.L.requires_grad = False

    def freeze_inducing(self):
        self.inducing_locations.requires_grad = False


class VaLLARegression(BaseVaLLA):
    def __init__(
        self,
        net,
        Z,
        prior_std,
        num_data,
        output_dim,
        log_variance=-5,
        track_inducing_locations=False,
        inducing_classes=None,
        y_mean=0.0,
        y_std=1.0,
        alpha=0.5,
        device=None,
        dtype=torch.float64,
    ):
        super().__init__(
            net,
            Z,
            prior_std,
            num_data,
            output_dim,
            track_inducing_locations,
            inducing_classes,
            y_mean,
            y_std,
            alpha,
            device,
            dtype,
        )

        self.log_variance = torch.tensor(log_variance, device=device, dtype=dtype)
        self.log_variance = torch.nn.Parameter(self.log_variance)

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

        mean, var = self.predict_y(predict_at)
        # Return predictions scaled to the original scale.
        return mean * self.y_std + self.y_mean, var * self.y_std**2

    def predict_y(self, X):
        mean, var = self.predict_f(X)
        return mean, var + self.log_variance.exp()

    def predict_f(self, X):
        """
        Performs the mean and covariance matrix of the given input.

        Parameters
        ----------
        X : torch tensor of shape (batch_size, data_dim)
            Contains the input features.

        Returns
        -------
        mean_pred : torch tensor of size (batch_size, output_dim)
                    Predictive mean of the model on the given batch
        var_pred : torch tensor of size (batch_size, output_dim, output_dim)
                   Contains the covariance matrix of the model for each element on
                   the batch.
        """

        # Transform flattened cholesky decomposition parameter into matrix
        L = torch.eye(self.num_inducing, dtype=self.dtype, device=self.device)
        li, lj = torch.tril_indices(self.num_inducing, self.num_inducing)
        # Shape (num_inducing, num_inducing)
        L[li, lj] = self.L

        F_mean, Kx_diag, Kxz, Kz = self.compute_kernels(X)
        self.Kz = Kz
        # L = torch.inverse(Kz) @ L

        # Compute auxiliar matrices
        # Shape [num_inducing, num_inducing]
        # H = I + L^T @ self.Kz @ L
        I = torch.eye(self.num_inducing, dtype=self.dtype, device=self.device)

        self.H = I + L.T @ self.Kz @ L

        # Shape [num_inducing, num_inducing]
        # A = L @ H^{-1} @ L^T
        self.A = L @ torch.linalg.solve(self.H, L.T)

        # Compute predictive diagonal
        # Shape [output_dim, output_dim, batch_size, batch_size]
        # K2 = Kxz @ A @ Kxz^T
        diag = torch.einsum("nma, ml, nlb -> nab", Kxz, self.A, Kxz)
        # Shape [batch_size, output_dim, output_dim]
        Fvar = Kx_diag - diag
        return F_mean, Fvar

    def logdensity(self, mu, var, x):
        """Computes the log density of a one dimensional Gaussian distribution
        of mean mu and variance var, evaluated on x.
        """
        logp = -0.5 * (np.log(2 * np.pi) + torch.log(var) + (mu - x) ** 2 / var)
        return logp

    def nelbo(self, X, y):
        """
        Computes the negative ELBO in the Hilbert space.

        Parameters
        ----------
        X : torch tensor of shape (batch_size, data_dim)
            Contains the input features.
        y : torch tensor of shape (batch_size, output_dim)
            Targets of the given input.

        Returns
        -------
        elbo : float
               The nelbo of the model at the current state for the given inputs.
        """
        F_mean, F_var = self.predict_f(X)
        if self.alpha == 0:
            logpdf = (
                -0.5 * np.log(2 * np.pi)
                - 0.5 * self.log_variance
                - 0.5
                * ((y - F_mean) ** 2 + F_var.squeeze(-1))
                / self.log_variance.exp()
            )
            logpdf = torch.sum(logpdf, -1)
        else:
            # Black-box alpha-energy
            variance = torch.exp(self.log_variance)
            # Proportionality constant
            C = (
                torch.sqrt(2 * torch.pi * variance / self.alpha)
                / torch.sqrt(2 * torch.pi * variance) ** self.alpha
            )

            logpdf = self.logdensity(
                F_mean, F_var.squeeze(-1) + variance / self.alpha, y
            )
            logpdf = logpdf + torch.log(C)
            logpdf = logpdf / self.alpha

        # # Aggregate on data dimension
        logpdf = torch.sum(logpdf)
        # Scale loss term corresponding to minibatch size
        scale = self.num_data
        scale /= X.shape[0]

        # Compute KL term
        KL = self.compute_KL()

        self.ell_history.append((-scale * logpdf).detach().cpu().numpy())
        self.kl_history.append(KL.detach().cpu().numpy())

        return -scale * logpdf + KL

    def predict_mean_and_var(self, X):
        """
        Computes the Predictive mean and variance of the model using the likelihood.

        Parameters
        ----------
        X : torch tensor of shape (batch_size, data_dim)
            Contains the input features.

        Returns
        -------
        mean_pred : torch tensor of size (batch_size, output_dim)
                    Predictive mean of the model on the given batch
        var_pred : torch tensor of size (batch_size, output_dim, output_dim)
                   Contains the covariance matrix of the model for each element on
                   the batch.
        """
        Fmu, F_var = self(X)
        return Fmean, Fvar + self.log_variance.exp()


class VaLLARegressionRBF(VaLLARegression):

    def __init__(
        self,
        net,
        Z,
        prior_std,
        num_data,
        output_dim,
        log_variance=-5,
        track_inducing_locations=False,
        inducing_classes=None,
        y_mean=0.0,
        y_std=1.0,
        alpha=0.5,
        device=None,
        dtype=torch.float64,
    ):
        super().__init__(
            net,
            Z,
            prior_std,
            num_data,
            output_dim,
            log_variance,
            track_inducing_locations,
            inducing_classes,
            y_mean,
            y_std,
            alpha,
            device,
            dtype,
        )


        self.length_scale = torch.ones(Z.shape[1], device=device, dtype=dtype)
        self.length_scale = torch.nn.Parameter(self.length_scale)

    def rbf(self, X, Z):
        X = X / self.length_scale**2
        Z = Z / self.length_scale**2

        dist = torch.sum((X.unsqueeze(1) - Z.unsqueeze(0)) ** 2, -1)

        l = 2
        K = self.prior_std**2 * torch.exp(-dist / l)
        return K

    def compute_kernels(self, X):
        Kx_diag = torch.diagonal(self.rbf(X, X)).unsqueeze(-1).unsqueeze(-1)
        Kxz = self.rbf(X, self.inducing_locations).unsqueeze(-1)
        Kzz = self.rbf(self.inducing_locations, self.inducing_locations)

        F_mean = self.net(X)

        return F_mean, Kx_diag, Kxz, Kzz


class VaLLAMultiClass(BaseVaLLA):
    def __init__(
        self,
        net_forward,
        Z,
        prior_std,
        num_data,
        n_classes_subsampled,
        output_dim,
        track_inducing_locations=False,
        fix_inducing_locations=False,
        inducing_classes=None,
        y_mean=0.0,
        y_std=1.0,
        device=None,
        dtype=torch.float64,
        seed=2147483647,
    ):
        super().__init__(
            net_forward,
            Z,
            prior_std,
            num_data,
            output_dim,
            track_inducing_locations,
            fix_inducing_locations,
            inducing_classes,
            y_mean,
            y_std,
            device,
            dtype,
            seed,
        )

        if n_classes_subsampled == -1:
            self.n_classes_sub_sampled = output_dim - 1
        else:
            self.n_classes_sub_sampled = n_classes_subsampled
        self.generator = torch.Generator(device=device)
        self.generator.manual_seed(seed)

    def compute_inducing_term(self, Kz):
        # Transform flattened cholesky decomposition parameter into matrix
        L = torch.eye(self.num_inducing, dtype=self.dtype, device=self.device)
        li, lj = torch.tril_indices(self.num_inducing, self.num_inducing)
        # Shape (num_inducing, num_inducing)
        L[li, lj] = self.L

        # Compute auxiliar matrices
        # Shape [num_inducing, num_inducing]
        # H = I + L^T @ self.Kz @ L
        I = torch.eye(self.num_inducing, dtype=self.dtype, device=self.device)
        self.H = I + torch.einsum("mn, ml, lk -> nk", L, Kz, L)
        # Shape [num_inducing, num_inducing]
        # A = L @ H^{-1} @ L^T
        # self.A = torch.einsum("nm, ml, kl -> nk", L, torch.inverse(self.H), L)
        self.A = L @ torch.linalg.solve(self.H, L.T)

    def compute_kernels(self, X, classes):
        Kxx_diagonal, Kxz, Kzz, piT_Kxx_pi, piT_Kxz, pi = self.net.NTK(
            X, self.inducing_locations, classes, self.inducing_class
        )
        Kxx_diagonal = self.prior_std**2 * Kxx_diagonal
        Kxz = self.prior_std**2 * Kxz
        Kzz = self.prior_std**2 * Kzz
        piT_Kxx_pi = self.prior_std**2 * piT_Kxx_pi
        piT_Kxz = self.prior_std**2 * piT_Kxz

        return Kxx_diagonal, Kxz, Kzz, piT_Kxx_pi, piT_Kxz, pi

    def nelbo(self, X, y):
        # Get clases
        F_mean = self.net(X)
        max_class = torch.argmax(F_mean, 1).unsqueeze(-1)
        all = torch.arange(0, self.output_dim).repeat((max_class.shape[0], 1))
        others = all.masked_fill(all == max_class, -1)
        mask = (others != -1).to(torch.float32)
        chosen = torch.multinomial(
            mask,
            num_samples=self.n_classes_sub_sampled,
            replacement=False,
            generator=self.generator,
        )
        classes = torch.concat([max_class, chosen], dim=-1).to(torch.long)

        # classes = torch.tile(torch.arange(0, 10, 1).unsqueeze(0), [classes.shape[0], 1])

        Kxx_diagonal, Kxz, Kzz, piT_Kxx_pi, piT_Kxz, pi = self.compute_kernels(
            X, classes
        )
        # Computes inducing term. It is stored in self.A
        self.compute_inducing_term(Kzz)
        self.Kz = Kzz

        K2 = torch.einsum("nm, ml, nl -> n", piT_Kxz, self.A, piT_Kxz)

        pi_Fvar_pi = piT_Kxx_pi - K2

        Fvar = Kxx_diagonal - torch.einsum("nma, ml, nla -> na", Kxz, self.A, Kxz)

        pi_subset = torch.gather(pi, 1, classes)
        scale = 1 / torch.sum(pi, -1)
        trace_pi_Fvar = scale * torch.sum(pi_subset * Fvar, -1)

        log_p = -trace_pi_Fvar + pi_Fvar_pi

        # Aggregate on data dimension
        ell = torch.sum(log_p)
        # Scale loss term corresponding to minibatch size
        scale = self.num_data
        scale /= X.shape[0]

        # Compute KL term
        KL = self.compute_KL()

        self.ell_history.append((-scale * ell).detach().cpu().numpy())
        self.kl_history.append(KL.detach().cpu().numpy())
        return -scale * ell + KL

    def forward(self, X):
        x, Kx, Kxz, Kzz = self.net.get_full_kernels(
            X, self.inducing_locations, self.inducing_class
        )

        # Computes inducing term. It is stored in self.A
        self.compute_inducing_term(self.prior_std**2 * Kzz)

        # Compute predictive diagonal
        # Shape [output_dim, output_dim, batch_size, batch_size]
        # K2 = Kxz @ A @ Kxz^T
        K2 = torch.einsum(
            "amb, ml, alk -> abk",
            self.prior_std**2 * Kxz,
            self.A,
            self.prior_std**2 * Kxz,
        )

        # Shape [batch_size, output_dim, output_dim]
        Fvar = self.prior_std**2 * Kx - K2

        return x, Fvar