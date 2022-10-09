from stat import FILE_ATTRIBUTE_VIRTUAL
import torch
from .utils import reparameterize, load_weights, extract_weights
import numpy as np
from .flows import *

import copy
from functorch import jacrev, jacfwd

class Test(torch.nn.Module):
    def __init__(
        self,
        prior_ip,
        variational_ip,
        likelihood,
        num_data,
        num_samples=1,
        bb_alpha=0,
        y_mean=0.0,
        y_std=1.0,
        device=None,
        dtype=torch.float64,
        seed=2147483647,
    ):
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

        self.generator = torch.Generator(device)
        self.generator.manual_seed(seed)

        # Set device and data type (precision)
        self.device = device
        self.dtype = dtype
        self.prior_ip = prior_ip
        self.variational_ip = variational_ip
        self.inducing_points = torch.tensor(np.linspace(-2, 2, 100)).unsqueeze(-1)

        self.bb_alphas = []
        self.KLs = []

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

    def kernel(self, x, y):
        return torch.exp(-torch.sum((x - y) ** 2, dim=[1]) / 2)

    def compute_reg(self):
        F = self.variational_ip(self.inducing_points)

        Fprior = self.prior_ip(self.inducing_points)

        m, n, d = F.shape

        k1 = self.kernel(F, F.transpose(0, 2))
        k2 = self.kernel(F, Fprior.transpose(0, 2))
        k3 = self.kernel(Fprior, Fprior.transpose(0, 2))

        return torch.sqrt(torch.mean(k1 - 2 * k2 + k3))

    def compute_reg(self):
        F = self.variational_ip(self.inducing_points)

        Fprior = self.prior_ip(self.inducing_points)

        m0 = torch.mean(F, dim=0)
        m1 = torch.mean(Fprior, dim=0)
        std0 = torch.std(F, dim=0) * np.sqrt(F.shape[0] / (F.shape[0] - 1))
        std1 = torch.std(Fprior, dim=0) * np.sqrt(
            Fprior.shape[0] / (Fprior.shape[0] - 1)
        )
        m = m1 - m0

        KL = -F.shape[1]
        KL += torch.sum((std0 / std1) ** 2)
        KL += torch.sum((m / std1) ** 2)
        KL += 2 * torch.log(torch.prod(std1))
        KL -= 2 * torch.log(torch.prod(std0))

        return 0.5 * KL

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
        F = self.variational_ip(X)

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
        KL = self.compute_reg()
        # KL = self.variational_ip.KL()

        self.bb_alphas.append((-scale * ve).detach().numpy())
        self.KLs.append(KL.detach().numpy())
        # print(-scale * ve)
        # print(KL)
        return -scale * ve + KL

    def forward(self, predict_at, num_samples=None):
        # Cast types if needed.
        if self.dtype != predict_at.dtype:
            predict_at = predict_at.to(self.dtype)

        mean, var = self.predict_y(predict_at, num_samples)
        # Return predictions scaled to the original scale.
        return mean * self.y_std + self.y_mean, torch.sqrt(var) * self.y_std

    def predict_f(self, predict_at, num_samples=None):

        F = self.variational_ip(predict_at)

        return F, torch.zeros_like(F)

    def predict_y(self, predict_at, num_samples=None):

        Fmean, Fvar = self.predict_f(predict_at, num_samples)
        return self.likelihood.predict_mean_and_var(Fmean, Fvar)

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

        F = self.prior_ip(X)

        # Scale data
        return F * self.y_std + self.y_mean

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

    def freeze_ll_variance(self):
        """Sets the likelihood variance as a non-trainable parameter."""
        self.likelihood.log_variance.requires_grad = False


class Test(torch.nn.Module):
    def __init__(
        self,
        prior_ip,
        variational_ip,
        Z,
        likelihood,
        num_data,
        num_samples=1,
        bb_alpha=0,
        y_mean=0.0,
        y_std=1.0,
        device=None,
        dtype=torch.float64,
        seed=2147483647,
    ):
        super().__init__()
        # Store data information
        self.num_data = num_data
        # Store Black-Box alpha value
        self.bb_alpha = bb_alpha

        self.inducing_points = torch.tensor(Z, dtype=dtype, device=device)
        # self.inducing_points = torch.nn.Parameter(self.inducing_points)

        # Store targets mean and std.
        self.y_mean = torch.tensor(y_mean, device=device)
        self.y_std = torch.tensor(y_std, device=device)

        # Set the amount of MC samples in training and test
        self.num_samples = num_samples

        # Store likelihood and Variational Implicit layers
        self.likelihood = likelihood

        self.generator = torch.Generator(device)
        self.generator.manual_seed(seed)

        # Set device and data type (precision)
        self.device = device
        self.dtype = dtype
        self.prior_ip = prior_ip
        self.variational_ip = variational_ip

        self.bb_alphas = []
        self.KLs = []

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

        F_mean, F_var, u, Pu_mean, Pu_var = self.predict_f(X)
        Qu_mean, Qu_var = self.gaussianize_taylor(self.variational_ip, self.inducing_points)
        # Qu_mean, Qu_var = self.gaussianize_samples(u)

        ve = self.likelihood.variational_expectations(
            F_mean, F_var, y, alpha=self.bb_alpha
        )

        # Aggregate on data dimension
        ve = torch.sum(ve)
        # Scale loss term corresponding to minibatch size
        scale = self.num_data
        scale /= X.shape[0]

        # Compute KL term
        KL = self.KL(Qu_mean, Qu_var, Pu_mean, Pu_var)

        self.bb_alphas.append((-scale * ve).cpu().detach().numpy())
        self.KLs.append(KL.cpu().detach().numpy())
        # print(-scale * ve)
        # print(KL)
        return -scale * ve + KL

    def forward(self, predict_at, num_samples=None):
        # Cast types if needed.
        if self.dtype != predict_at.dtype:
            predict_at = predict_at.to(self.dtype)

        mean, var, u = self.predict_y(predict_at, num_samples)
        # Return predictions scaled to the original scale.
        return (
            mean * self.y_std + self.y_mean,
            torch.sqrt(var) * self.y_std,
            # u * self.y_std + self.y_mean,
        )

    def gaussianize_samples(self, f):
        mean = torch.mean(f, dim=0)
        m = f - mean
        cov = torch.einsum("snd, smd ->nmd", m, m) / (f.shape[0] - 1)
        return mean, cov

    def gaussianize_taylor(self, ip, X):

        w = ip.get_weights()
        J = jacrev(ip.forward_weights, argnums = 1)(X, w)

        J = torch.cat([torch.flatten(a, -2, -1) for a in J], dim=-1).transpose(-1, -2)
        S = torch.exp(ip.get_std_params()) ** 2

        cov = J * S.unsqueeze(0).unsqueeze(-1)
        cov = torch.einsum("nsd, msd -> nmd", cov, J)
        mean = ip.forward_mean(X)
        return mean, cov

    # def KL(self, Qu, Pu_mean, Pu_var, Z=None):
    #     # Needs testing, seems to be wrong
    #     Qu_mean, Qu_var = self.gaussianize2(Qu, prior=self.variational_ip, X=Z)

    #     inv = torch.inverse(
    #         Pu_var.transpose(0, -1) + 1e-6 * torch.eye(Pu_var.shape[0])
    #     ).transpose(0, -1)

    #     KL = -Pu_mean.shape[0]
    #     # Trace
    #     KL += torch.sum(inv * Qu_var.transpose(0, 1), dim=[0, 1])
    #     # Quadratic term
    #     KL += torch.einsum(
    #         "nd, nmd, md -> d", Pu_mean - Qu_mean, inv, Pu_mean - Qu_mean
    #     )

    #     KL += torch.log(torch.det(Pu_var.transpose(0, -1)))
    #     KL -= torch.log(torch.det(Qu_var.transpose(0, -1)))
    #     return 0.5 * torch.sum(KL)

    def KL(self, mu1, var1, mu2, var2):
        var1 = torch.diagonal(var1).T
        var2 = torch.diagonal(var2).T

        inv = torch.inverse(
            var2.transpose(0, -1) + 1e-6 * torch.eye(var2.shape[0])
        ).transpose(0, -1)

        KL = -mu2.shape[0]
        # Trace
        KL += torch.sum(var1 / var2, dim=[0])
        # Quadratic term
        KL += torch.sum((mu2 - mu1) ** 2 / var2, dim=0)

        KL += torch.log(torch.prod(var2, dim=0))
        KL -= torch.log(torch.prod(var1, dim=0))
        return 0.5 * torch.sum(KL)

    def generate_u_samples(self):
        return self.variational_ip(self.inducing_points)

    def predict_f(self, X, num_samples=None):

        # Batch size
        n = X.shape[0]
        # Concatenation of batch and inducing points
        X_and_Z = torch.concat([X, self.inducing_points], axis=0)
        # f([X, Z])

        # Compute P(f([X, Z]))
        mean, cov = self.gaussianize_taylor(self.prior_ip, X_and_Z)
        # F = self.prior_ip(X_and_Z)
        # mean, cov = self.gaussianize_samples(F)

        # Sample from Q(u)
        u = self.generate_u_samples()

        # P(u) mean and covariance matrix
        Pu_mean = mean[n:]
        Pu_var = cov[n:, n:]

        # Compute inverse of P(u) variance
        inv = torch.inverse(
            Pu_var.transpose(0, -1) + 1e-6 * torch.eye(Pu_var.shape[0])
        ).transpose(0, -1)

        # Auxiliar matrix
        A = torch.einsum("abd, bcd -> acd", cov[:n, n:], inv)

        # Compute mean of P(f|u)
        Pfu_mean = mean[:n] + torch.einsum("abd, sbd -> sad", A, u - mean[n:])

        # Compute diagonal of P(f|u), variance does not depend on the value of u
        Pfu_var = torch.diagonal(cov[:n, :n]).T - torch.einsum(
            "abd, bad -> ad", A, cov[n:, :n]
        )
        # Replicate variance for every sample of Q(u)
        Pfu_var = torch.tile(Pfu_var.unsqueeze(0), (Pfu_mean.shape[0], 1, 1))

        return Pfu_mean, Pfu_var, u, Pu_mean, Pu_var

    def predict_y(self, predict_at, num_samples=None):

        Fmean, Fvar, Qu, _, _ = self.predict_f(predict_at, num_samples)
        return *self.likelihood.predict_mean_and_var(Fmean, Fvar), Qu

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

        F = self.prior_ip(X)

        # Scale data
        return F * self.y_std + self.y_mean

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

    def freeze_ll_variance(self):
        """Sets the likelihood variance as a non-trainable parameter."""
        self.likelihood.log_variance.requires_grad = False


class Test2(Test):
    def __init__(
        self,
        prior_ip,
        variational_ip,
        Z,
        likelihood,
        num_data,
        num_samples=1,
        bb_alpha=0,
        y_mean=0.0,
        y_std=1.0,
        device=None,
        dtype=torch.float64,
        seed=2147483647,
    ):
        super().__init__(
            prior_ip,
            variational_ip,
            Z,
            likelihood,
            num_data,
            num_samples,
            bb_alpha,
            y_mean,
            y_std,
            device,
            dtype,
            seed,
        )

        self.flow = CouplingFlow(5, Z.shape[0], device, dtype, seed=2147483647)

    def predict_f(self, X, num_samples=None):
        n = X.shape[0]
        i = torch.concat([X, self.inducing_points], axis=0)
        F = self.prior_ip(i)
        mean, cov = self.gaussianize(F, i)

        Pu_mean = mean[n:]
        Pu_var = cov[n:, n:]
        inv = torch.inverse(Pu_var + 1e-6 * np.eye(Pu_var.shape[0]))
        Qu = self.generate_u_samples(F[:, n:])

        Pfu_mean = mean[:n] + cov[:n, n:] @ inv @ (Qu - mean[n:])
        Pfu_var = cov[:n, :n] - cov[:n, n:] @ inv @ cov[:n, n:].T

        F_mean = Pfu_mean
        Fvar = torch.tile(
            torch.diagonal(Pfu_var).unsqueeze(-1).unsqueeze(0), (F_mean.shape[0], 1, 1)
        )
        return F_mean, Fvar, Qu, Pu_mean, Pu_var

    def generate_u_samples(self, u):
        u, LDJ = self.flow(u.reshape(u.shape[0], -1), self.inducing_points)
        self.LDJ = LDJ
        return u.unsqueeze(-1)

    def KL(self, *args):
        KL = torch.mean(self.LDJ * torch.exp(self.LDJ))
        return KL
