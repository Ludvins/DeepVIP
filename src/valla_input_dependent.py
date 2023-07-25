import torch
from .utils import reparameterize
import numpy as np
import torch.autograd.profiler as profiler
from torch import nn


def init_weights(m):
    if isinstance(m, nn.Linear):
        n = m.in_features
        y = 1.0 / np.sqrt(n)
        m.weight.data.uniform_(-y, y)
        m.bias.data.fill_(0)


class NeuralNetwork(torch.nn.Module):
    def __init__(self, num_inducing, input_dim, inner_dims, device, dtype):
        super(NeuralNetwork, self).__init__()

        torch.manual_seed(2147483647)
        layers = []
        dims = [input_dim] + inner_dims

        self.input_dim = input_dim
        self.num_inducing = num_inducing

        for i, (_in, _out) in enumerate(zip(dims[:-1], dims[1:])):
            print(_in, _out)
            layers.append(torch.nn.Linear(_in, _out, device=device, dtype=dtype))
            # layers.append(torch.nn.BatchNorm1d(_out, dtype = dtype, device = device)),
            layers.append(torch.nn.Tanh())

        output_dim_cholesky = int(num_inducing * (num_inducing + 1) / 2)
        self.feature_extractor = torch.nn.Sequential(*layers)

        # self.feature_extractor.apply(init_weights)
        self.inducing_locations = nn.Linear(
            in_features=_out,
            out_features=num_inducing * input_dim,
            device=device,
            dtype=dtype,
        )
        # self.inducing_locations.apply(init_weights)
        self.cholesky_factor = nn.Linear(
            in_features=_out,
            out_features=output_dim_cholesky,
            device=device,
            dtype=dtype,
        )
        # self.cholesky_factor.apply(init_weights)

    def forward(self, x):
        fe = self.feature_extractor(x)
        input_loc = self.inducing_locations(fe)
        input_loc = input_loc.reshape([x.shape[0], self.num_inducing, self.input_dim])
        input_loc = input_loc  # + x.unsqueeze(1)
        cf = self.cholesky_factor(fe)
        return input_loc, cf


class BaseVaLLA(torch.nn.Module):
    def __init__(
        self,
        num_inducing,
        inducing_net_size,
        net_forward,
        prior_std,
        num_data,
        input_dim,
        output_dim,
        y_mean=0.0,
        y_std=1.0,
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

        self.net = net_forward

        self.prior_std = torch.tensor(prior_std, device=device, dtype=dtype)

        self.num_inducing = num_inducing

        self.inducing_net = NeuralNetwork(
            self.num_inducing, input_dim, inducing_net_size, device=device, dtype=dtype
        )

        # Set device and data type (precision)
        self.device = device
        self.dtype = dtype

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

    def forward(self, X):
        raise NotImplementedError

    def compute_KL(self):
        """
        Computes the Kulback-Leibler divergence between the variational distribution
        and the prior.
        """

        log_det = torch.logdet(self.H)

        trace = torch.sum(torch.diagonal(self.Kz @ self.A, dim1=-2, dim2=-1), -1)
        KL = 0.5 * log_det - 0.5 * trace
        return torch.mean(KL)

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

    def freeze_ll(self):
        for param in self.likelihood.parameters():
            param.requires_grad = False


class VaLLARegression(BaseVaLLA):
    def __init__(
        self,
        num_inducing,
        inducing_net_size,
        net_forward,
        prior_std,
        likelihood,
        num_data,
        input_dim,
        output_dim,
        y_mean=0.0,
        y_std=1.0,
        device=None,
        dtype=torch.float64,
    ):
        super().__init__(
            num_inducing,
            inducing_net_size,
            net_forward,
            prior_std,
            num_data,
            input_dim,
            output_dim,
            y_mean,
            y_std,
            device,
            dtype,
        )

        # Store likelihood and Variational Implicit layers
        self.likelihood = likelihood
        self.inducing_class = torch.zeros(num_inducing, device=device, dtype=torch.long)

    def compute_kernels(self, X, Z):
        F_mean, Kx_diag, Kxz, Kz = self.net.get_full_kernels_dependent(
            X, Z, self.inducing_class
        )
        Kx_diag = self.prior_std**2 * Kx_diag
        Kxz = self.prior_std**2 * Kxz
        Kz = self.prior_std**2 * Kz

        return F_mean, Kx_diag, Kxz, Kz

    def forward(self, X):
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

        Z, tril = self.inducing_net(X)

        li, lj = torch.tril_indices(self.num_inducing, self.num_inducing)

        L = torch.eye(
            self.num_inducing, dtype=self.dtype, device=self.device
        ).unsqueeze(0)
        L = torch.tile(L, [tril.shape[0], 1, 1])
        L[:, li, lj] = tril

        F_mean, Kx_diag, Kxz, Kz = self.compute_kernels(X, Z)
        # Clean gradients of inducing locations
        self.Kz = Kz

        # Compute auxiliar matrices
        # Shape [num_inducing, num_inducing]
        # H = I + L^T @ self.Kz @ L
        I = torch.eye(self.num_inducing, dtype=self.dtype, device=self.device)

        self.H = I + L.permute(0, 2, 1) @ self.Kz @ L

        # Shape [num_inducing, num_inducing]
        # A = L @ H^{-1} @ L^T
        self.A = L @ torch.linalg.solve(self.H, L.permute(0, 2, 1))

        # Compute predictive diagonal
        # Shape [output_dim, output_dim, batch_size, batch_size]
        # K2 = Kxz @ A @ Kxz^T
        diag = torch.einsum("nma, nml, nlb -> nab", Kxz, self.A, Kxz)

        # Shape [batch_size, output_dim, output_dim]
        Fvar = Kx_diag - diag
        return F_mean, Fvar

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
        F_mean, F_var = self(X)

        bb_alpha = self.likelihood.variational_expectations(F_mean, F_var, y, alpha=0)

        # Aggregate on data dimension
        bb_alpha = torch.sum(bb_alpha)

        # Scale loss term corresponding to minibatch size
        scale = self.num_data
        scale /= X.shape[0]

        # Compute KL term
        KL = self.compute_KL()

        self.ell_history.append((-scale * bb_alpha).detach().cpu().numpy())
        self.kl_history.append(KL.detach().cpu().numpy())

        return -scale * bb_alpha + KL


class VaLLAMultiClass(BaseVaLLA):
    def __init__(
        self,
        num_inducing,
        inducing_net_size,
        net_forward,
        n_classes_subsampled,
        prior_std,
        num_data,
        input_dim,
        output_dim,
        y_mean=0.0,
        y_std=1.0,
        device=None,
        dtype=torch.float64,
        seed=2147483647,
    ):
        super().__init__(
            num_inducing,
            inducing_net_size,
            net_forward,
            prior_std,
            num_data,
            input_dim,
            output_dim,
            y_mean,
            y_std,
            device,
            dtype,
        )

        # Store likelihood and Variational Implicit layers
        #
        if n_classes_subsampled == -1:
            self.n_classes_sub_sampled = output_dim - 1
        else:
            self.n_classes_sub_sampled = n_classes_subsampled

        if num_inducing < output_dim:
            raise ValueError("Num inducing has to be greater than outut dim")

        self.inducing_class = torch.tensor(
            np.tile(
                np.arange(self.output_dim),
                reps=np.ceil(self.num_inducing / self.output_dim).astype(int),
            )[: self.num_inducing],
            device=device,
            dtype=torch.long,
        )

        self.generator = torch.Generator(device=device)
        self.generator.manual_seed(seed)

    def compute_inducing_term(self, Kz, tril):
        li, lj = torch.tril_indices(self.num_inducing, self.num_inducing)

        L = torch.eye(
            self.num_inducing, dtype=self.dtype, device=self.device
        ).unsqueeze(0)
        L = torch.tile(L, [tril.shape[0], 1, 1])
        L[:, li, lj] = tril

        # Compute auxiliar matrices
        # Shape [num_inducing, num_inducing]
        # H = I + L^T @ self.Kz @ L
        I = torch.eye(self.num_inducing, dtype=self.dtype, device=self.device)

        self.H = I + L.permute(0, 2, 1) @ Kz @ L

        # Shape [num_inducing, num_inducing]
        # A = L @ H^{-1} @ L^T
        self.A = L @ torch.linalg.solve(self.H, L.permute(0, 2, 1))

    def compute_kernels(self, X, Z, classes):
        Kxx_diagonal, Kxz, Kzz, piT_Kxx_pi, piT_Kxz, pi = self.net.NTK_dependent(
            X, Z, classes, self.inducing_class
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

        Z, tril = self.inducing_net(X)

        Kxx_diagonal, Kxz, Kzz, piT_Kxx_pi, piT_Kxz, pi = self.compute_kernels(
            X, Z, classes
        )

        # Computes inducing term. It is stored in self.A
        self.compute_inducing_term(Kzz, tril)
        self.Kz = Kzz

        K2 = torch.einsum("nm, nml, nl -> n", piT_Kxz, self.A, piT_Kxz)

        pi_Fvar_pi = piT_Kxx_pi - K2

        Fvar = Kxx_diagonal - torch.einsum("nma, nml, nla -> na", Kxz, self.A, Kxz)

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
        Z, tril = self.inducing_net(X)

        x, Kx, Kxz, Kzz = self.net.get_full_kernels_dependent(X, Z, self.inducing_class)

        # Computes inducing term. It is stored in self.A
        self.compute_inducing_term(self.prior_std**2 * Kzz, tril)

        # Compute predictive diagonal
        # Shape [output_dim, output_dim, batch_size, batch_size]
        # K2 = Kxz @ A @ Kxz^T
        K2 = torch.einsum(
            "amb, aml, alk -> abk",
            self.prior_std**2 * Kxz,
            self.A,
            self.prior_std**2 * Kxz,
        )

        # Shape [batch_size, output_dim, output_dim]
        Fvar = self.prior_std**2 * Kx - K2

        return x, Fvar
