#!/usr/bin/env python3
import torch
import numpy


class GPLLA(torch.nn.Module):
    def name(self):
        return "SparseLA"

    def __init__(
        self,
        net_forward,
        prior_std,
        likelihood_hessian,
        likelihood,
        backend,
        device=None,
        dtype=torch.float64,
    ):
        super().__init__()

        self.backend = backend
        self.net = net_forward

        self.prior_std = torch.tensor(prior_std, device=device, dtype=dtype)

        self.likelihood_hessian = likelihood_hessian
        self.likelihood = likelihood

        # Set device and data type (precision)
        self.device = device
        self.dtype = dtype

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

        return 0, Fmean, Fvar

    def jacobian_features(self, X):
        Js, _ = self.backend.jacobians(x=X)
        return Js * self.prior_std

    def fit(self, X_train, y_train):
        # Shape (num_data, output_dim, num_parameters)
        self.Jx = self.jacobian_features(X_train)

        # Shape (output_dim, output_dim, num_data, num_data)
        Kx = torch.einsum("nas, mbs -> anbm", self.Jx, self.Jx)
        output_dim = Kx.shape[0]
        n_data = Kx.shape[1]

        # Shape (output_dim, output_dim, num_data, num_data)
        Lambda = self.likelihood_hessian(X_train, y_train)
        Lambda = torch.diag_embed(Lambda)

        Lambda = (
            Lambda.permute(0, 2, 1, 3)
            .flatten(start_dim=0, end_dim=1)
            .flatten(start_dim=1, end_dim=2)
        )
        Lambda_inv = torch.inverse(Lambda + 1e-7 * torch.eye(Lambda.shape[0]))
        Lambda_inv = Lambda_inv.unflatten(0, (output_dim, n_data)).unflatten(
            2, (output_dim, n_data)
        )
        # Shape (output_dim, output_dim, num_data, num_data)
        K = Kx + Lambda_inv

        K = K.flatten(start_dim=0, end_dim=1).flatten(start_dim=1, end_dim=2)
        K_inv = torch.inverse(K + 1e-7 * torch.eye(K.shape[0]))
        K_inv = K_inv.unflatten(0, (output_dim, n_data)).unflatten(
            2, (output_dim, n_data)
        )
        self.inv = K_inv.permute(0, 2, 1, 3)

    def forward(self, X):
        # Shape (batch_size, output_dim)
        mean = self.net(X)

        # Shape (bath_size, output_dim, num_parameters)
        Jz = self.jacobian_features(X)

        # Shape (output_dim, output_dim, batch_size)
        Kzz = torch.einsum("nas, nbs -> abn", Jz, Jz)

        # Shape (output_dim, output_dim, batch_size, num_data)
        Kzx = torch.einsum("nas, mbs -> abnm", Jz, self.Jx)

        K2 = torch.einsum("abnm, bcml, dckl -> adnk", Kzx, self.inv, Kzx)

        # Shape (output_dim, output_dim, batch_size)
        KLLA = Kzz - torch.diagonal(K2, dim1=-2, dim2=-1)

        # Permute variance to have shape ( num_data, output_dim, output_dim)
        return mean, KLLA.permute(2, 0, 1)

    def predict_mean_and_var(self, X):
        Fmu, F_var = self(X)
        return self.likelihood.predict_mean_and_var(Fmu, F_var)


class GPLLA_Optimized(torch.nn.Module):
    def name(self):
        return "SparseLA"

    def __init__(
        self,
        net,
        prior_std,
        likelihood_hessian,
        likelihood,
        device=None,
        dtype=torch.float64,
    ):
        super().__init__()

        self.net = net

        self.prior_std = torch.tensor(prior_std, device=device, dtype=dtype)

        self.likelihood_hessian = likelihood_hessian
        self.likelihood = likelihood

        # Set device and data type (precision)
        self.device = device
        self.dtype = dtype

    def fit(self, X_train, y_train):
        self.X_train = X_train
        # Shape (output_dim, output_dim, num_data, num_data)
        Kx = self.net.get_kernel(X_train, X_train).permute(2, 0, 3, 1)

        output_dim = Kx.shape[0]
        n_data = Kx.shape[1]

        # Shape (output_dim, output_dim, num_data, num_data)
        Lambda = self.likelihood_hessian(X_train, y_train)
        Lambda = torch.diag_embed(Lambda)

        Lambda = (
            Lambda.permute(0, 2, 1, 3)
            .flatten(start_dim=0, end_dim=1)
            .flatten(start_dim=1, end_dim=2)
        )
        Lambda_inv = torch.inverse(Lambda + 1e-7 * torch.eye(Lambda.shape[0]))
        Lambda_inv = Lambda_inv.unflatten(0, (output_dim, n_data)).unflatten(
            2, (output_dim, n_data)
        )
        # Shape (output_dim, output_dim, num_data, num_data)
        K = Kx + Lambda_inv / (self.prior_std**2)

        K = K.flatten(start_dim=0, end_dim=1).flatten(start_dim=1, end_dim=2)
        K_inv = torch.inverse(K + 1e-7 * torch.eye(K.shape[0]))
        K_inv = K_inv.unflatten(0, (output_dim, n_data)).unflatten(
            2, (output_dim, n_data)
        )
        self.inv = K_inv.permute(0, 2, 1, 3)

    def forward(self, X):
        # Shape (batch_size, output_dim)
        mean = self.net(X)

        # Shape (output_dim, output_dim, batch_size)
        Kzz = self.net.get_kernel(X, X)
        Kzz = torch.diagonal(Kzz, dim1=0, dim2=1).permute(2, 0, 1)

        # Shape (output_dim, output_dim, batch_size, num_data)
        Kzx = self.net.get_kernel(X, self.X_train).permute(2, 3, 0, 1)

        K2 = torch.einsum("abnm, bcml, dcnl -> nad", Kzx, self.inv, Kzx)
        # Shape (output_dim, output_dim, batch_size)
        KLLA = Kzz - K2

        # Permute variance to have shape ( num_data, output_dim, output_dim)
        return mean, self.prior_std**2 * KLLA

    def predict_mean_and_var(self, X):
        Fmu, F_var = self(X)
        return self.likelihood.predict_mean_and_var(Fmu, F_var)
