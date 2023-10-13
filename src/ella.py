#!/usr/bin/env python3

import torch
import numpy as np
from utils.pytorch_learning import score
from utils.metrics import SoftmaxClassificationNLL, RegressionNLL
from src.backpack_interface import BackPackInterface

class ELLA_Base(torch.nn.Module):
    def __init__(
        self,
        net,
        output_size,
        n_samples,
        n_eigh,
        prior_std,
        seed,
        y_mean,
        y_std,
        device=None,
        dtype=torch.float64,
    ):
        super().__init__()

        self.output_size = output_size
        self.net = net

        self.M = n_samples
        self.K = n_eigh
        self.y_mean = torch.tensor(y_mean, device=device)
        self.y_std = torch.tensor(y_std, device=device)

        self.seed = seed

        self.prior_std = torch.tensor(prior_std, device=device, dtype=dtype)

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
    
    def predict_f(self, X_test):

        G_inv = torch.inverse(self.G + torch.eye(self.K, dtype=self.dtype) / (self.prior_std**2))

        with torch.no_grad():

            Jz = self.net.get_jacobian(X_test)

            phi = torch.einsum("mds, sk -> dmk", Jz, self.v)

            K_test = torch.einsum("ank, kg, bmg -> abnm", phi, G_inv, phi)

            mean = self.net(X_test)
            var = torch.diagonal(K_test, dim1=-2, dim2=-1)

        return mean, var.permute(2, 0, 1)


    def fit_loader_val(self, X_train, y_train, train_loader, val_loader, val_steps=1, verbose = True):
        with torch.no_grad():
            rng = np.random.default_rng(self.seed)
            indexes = rng.choice(np.arange(X_train.shape[0]), self.M, replace=False)
            self.Xz = X_train[indexes]

            if self.output_size > 1:
                phi_z = self.net.get_jacobian_on_outputs(
                    self.Xz, y_train[indexes].to(torch.long).squeeze(-1)
                )
            else:
                phi_z = self.net.get_jacobian(self.Xz)

            phi_z = phi_z.squeeze(1)

            K = torch.einsum("ns, ms -> nm", phi_z, phi_z)

            L, V = torch.linalg.eigh(K)

            L = torch.abs(L[-self.K :]).flip(-1)
            V = V[:, -self.K :].flip(-1)

            self.v = torch.einsum("ms, mk -> sk", phi_z, V / torch.sqrt(L).unsqueeze(0))

            self.G = torch.zeros(self.K, self.K, dtype=self.dtype)

            if verbose:
                from tqdm import tqdm

                iters = tqdm(range(len(train_loader)), unit="iteration")
                iters.set_description("Training ")
            else:
                iters = np.arange(len(train_loader))

            data_iter = iter(train_loader)

            best_vall_nll = np.infty

            # Batches evaluation
            for i in iters:
                if i % val_steps == 0 and val_loader is not None:
                    
                    val_nll = score(
                            self,
                            val_loader,
                            self.val_metrics,
                            use_tqdm=verbose,
                            device=self.device,
                            dtype=self.dtype,
                        )

                    if val_nll < best_vall_nll:
                        best_vall_nll = val_nll
                    else:
                        print("Training ended at {} iterations".format(i))
                        break


                data, target = next(data_iter)
                data = data.to(self.device).to(self.dtype)
                target = target.to(self.device).to(self.dtype)

                Lambda = self.likelihood_hessian(data, target)

                Jtrain = self.net.get_jacobian(data)
                phi_train = torch.einsum("mds, sk -> dmk", Jtrain, self.v)
                self.G += torch.einsum("amk, abm, bmg -> kg", phi_train, Lambda, phi_train)


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


class ELLA_Regression(ELLA_Base):
    def __init__(
        self,
        net,
        output_size,
        n_samples,
        n_eigh,
        prior_std,
        seed,
        log_variance,
            y_mean, y_std,
        device=None,
        dtype=torch.float64,
    ):
        super().__init__(
            net,
            output_size,
            n_samples,
            n_eigh,
            prior_std,
            seed,
            y_mean, y_std,
            device=None,
            dtype=torch.float64,
        )
        self.log_variance = torch.tensor(log_variance, device = device, dtype = dtype)
        self.val_metrics = RegressionNLL

    def likelihood_hessian(self, data, target):
        ones = torch.ones_like(target).unsqueeze(-1).permute(1, 2, 0)
        return ones / self.log_variance.exp()

    def predict_y(self, X):
        mean, var = self.predict_f(X)
        return mean, var + self.log_variance.exp()


class ELLA_Multiclass(ELLA_Base):
    def __init__(
        self,
        net,
        output_size,
        n_samples,
        n_eigh,
        prior_std,
        seed,
        device=None,
        dtype=torch.float64,
    ):
        super().__init__(
            net,
            output_size,
            n_samples,
            n_eigh,
            prior_std,
            seed,
            0, 1,
            device=device,
            dtype=dtype,
        )
        self.val_metrics = SoftmaxClassificationNLL

    def likelihood_hessian(self, data, target):
        out = torch.nn.Softmax(dim=-1)(self.net(data))
        a = torch.einsum("na, nb -> abn", out, out)
        b = torch.diag_embed(out).permute(1, 2, 0)
        return -a + b

    def predict_y(self, X):
        return self.predict_f(X)
    


class ELLA_MulticlassBackpack(ELLA_Base):
    def __init__(
        self,
        net,
        output_size,
        n_samples,
        n_eigh,
        prior_std,
        seed,
        device=None,
        dtype=torch.float64,
    ):
        super().__init__(
            net,
            output_size,
            n_samples,
            n_eigh,
            prior_std,
            seed,
            0, 1,
            device=device,
            dtype=dtype,
        )
        self.backpack = BackPackInterface(net, output_size)
        self.val_metrics = SoftmaxClassificationNLL

    def likelihood_hessian(self, data, target):
        out = torch.nn.Softmax(dim=-1)(self.net(data))
        a = torch.einsum("na, nb -> abn", out, out)
        b = torch.diag_embed(out).permute(1, 2, 0)
        return -a + b
    

    def fit_loader_val(self, X_train, y_train, train_loader, val_loader, val_steps=1, verbose = True):
        with torch.no_grad():
            rng = np.random.default_rng(self.seed)
            indexes = rng.choice(np.arange(X_train.shape[0]), self.M, replace=False)
            self.Xz = X_train[indexes]

            if self.output_size > 1:
                phi_z = self.backpack.jacobians_on_outputs(
                    self.Xz, y_train[indexes].to(torch.long)
                )
            else:
                phi_z = self.backpack.jacobians(self.Xz)

            phi_z = phi_z.squeeze(1)

            K = torch.einsum("ns, ms -> nm", phi_z, phi_z)

            L, V = torch.linalg.eigh(K)

            L = torch.abs(L[-self.K :]).flip(-1)
            V = V[:, -self.K :].flip(-1)

            self.v = torch.einsum("ms, mk -> sk", phi_z, V / torch.sqrt(L).unsqueeze(0))

            self.G = torch.zeros(self.K, self.K, dtype=self.dtype)

            if verbose:
                from tqdm import tqdm

                iters = tqdm(range(len(train_loader)), unit="iteration")
                iters.set_description("Training ")
            else:
                iters = np.arange(len(train_loader))

            data_iter = iter(train_loader)

            best_vall_nll = np.infty

            # Batches evaluation
            for i in iters:
                if i % val_steps == 0:
                    
                    val_nll = score(
                            self,
                            val_loader,
                            self.val_metrics,
                            use_tqdm=verbose,
                            device=self.device,
                            dtype=self.dtype,
                        )

                    if val_nll < best_vall_nll:
                        best_vall_nll = val_nll
                    else:
                        print("Training ended at {} iterations".format(i))
                        break


                data, target = next(data_iter)
                data = data.to(self.device).to(self.dtype)
                target = target.to(self.device).to(self.dtype)

                Lambda = self.likelihood_hessian(data, target)

                Jtrain = self.backpack.jacobians(data)
                phi_train = torch.einsum("mds, sk -> dmk", Jtrain, self.v)
                self.G += torch.einsum("amk, abm, bmg -> kg", phi_train, Lambda, phi_train)

    def predict_f(self, X_test):
        with torch.no_grad():
            G_inv = torch.inverse(self.G + torch.eye(self.K, dtype=self.dtype) / (self.prior_std**2))

            Jz = self.backpack.jacobians(X_test)

            phi = torch.einsum("mds, sk -> dmk", Jz, self.v)

            K_test = torch.einsum("ank, kg, bmg -> abnm", phi, G_inv, phi)

            mean = self.net(X_test)
            var = torch.diagonal(K_test, dim1=-2, dim2=-1)

            return mean, var.permute(2, 0, 1)


    def predict_y(self, X):
        return self.predict_f(X)