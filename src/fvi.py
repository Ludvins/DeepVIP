import torch
import numpy as np
from .flows import *
from .noise_samplers import *

from functorch import jacrev, jacfwd, hessian


class FVI(torch.nn.Module):
    def __init__(
        self,
        prior_ip,
        variational_ip,
        Z,
        likelihood,
        num_data,
        fix_inducing,
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
        if not fix_inducing:
            self.inducing_points = torch.nn.Parameter(self.inducing_points)
            
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

        F_mean, F_var, Pu_mean, Pu_var = self.predict_f(X, self.num_samples)
        
        Qu_mean, Qu_var = self.gaussianize_taylor(
            self.variational_ip, self.inducing_points.detach()
        )

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
        if num_samples is None:
            num_samples = self.num_samples
        # Cast types if needed.
        if self.dtype != predict_at.dtype:
            predict_at = predict_at.to(self.dtype)

        mean, var = self.predict_y(predict_at, num_samples)
        # Return predictions scaled to the original scale.
        return (
            mean * self.y_std + self.y_mean,
            torch.sqrt(var) * self.y_std,
        )

    def gaussianize_taylor(self, ip, X):

        w = ip.get_weights()

        J = jacrev(ip.forward_weights, argnums=1)(X, w)
 
        J = (
            torch.cat(
                [
                    torch.flatten(a, 2, -1) if len(a.shape) > 2 else a.unsqueeze(-1)
                    for a in J
                ],
                dim=-1,
            )
            .transpose(-1, -2)
            .to(self.dtype)
        )
        H = hessian(ip.forward_weights, argnums = 1)(X, w)


        H = torch.cat([
                torch.cat([
                    H[i][j].flatten(-4, -3).flatten(-2, -1) 
                for j in range(len(H))], dim=-1)
            for i in range(len(H))], dim = -2)

        H = torch.diagonal(H, dim1=-2, dim2 = -1).transpose(-1, -2).to(self.dtype)
        
        S = torch.exp(ip.get_std_params()) ** 2

        cov = torch.einsum("nsd, s, msd -> nmd", J, S, J) #
        cov = cov + torch.einsum("nsd, s, msd -> nmd", H, S**2, H)
        mean = ip.forward_mean(X)
        mean = mean + 0.5 * torch.einsum("s, nsd-> nd",S, H)
        return mean, cov

    def KL(self, Qu_mean, Qu_var, Pu_mean, Pu_var):
        
        # d1 = torch.distributions.multivariate_normal.MultivariateNormal(
        #     loc = Qu_mean.transpose(0, -1),
        #     covariance_matrix = Qu_var.permute(2, 0, 1)
        # )
        # d2 = torch.distributions.multivariate_normal.MultivariateNormal(
        #     loc = Pu_mean.transpose(0, -1),
        #     covariance_matrix = Pu_var.permute(2, 0, 1)
        # )
        # KL =  torch.sum(torch.distributions.kl.kl_divergence(d1, d2))
        # return KL
        
        I = 1e-6 * torch.eye(Pu_var.shape[0])

        L0 = torch.linalg.cholesky(Qu_var.permute(2, 0, 1) + I)
        L1 = torch.linalg.cholesky(Pu_var.permute(2, 0, 1) + I)

        M = torch.linalg.solve_triangular(L1, L0, upper=False)
        M = torch.diagonal(M, dim1=-2, dim2=-1)

        m = (Pu_mean - Qu_mean).transpose(0, -1).unsqueeze(-1)
        y = torch.linalg.solve_triangular(L1, m, upper=False).squeeze(-1)
        L1 = torch.diagonal(L1, dim1=-2, dim2=-1)
        L0 = torch.diagonal(L0, dim1=-2, dim2=-1)

        KL = -Pu_mean.shape[0]

        KL += torch.sum(M ** 2, dim=-1)
        KL += torch.sum(y ** 2, dim=-1)

        KL += 2 * torch.sum(torch.log(L1), dim=-1)
        KL -= 2 * torch.sum(torch.log(L0), dim=-1)

        KL =  0.5 * torch.sum(KL)
        return KL

    def generate_u_samples(self, num_samples):
        u = self.variational_ip(self.inducing_points, num_samples)
        return u

    def predict_f(self, X, num_samples=None):

        # Batch size
        n = X.shape[0]
        # Concatenation of batch and inducing points
        X_and_Z = torch.concat([X, self.inducing_points], axis=0)
        # Compute P(f([X, Z]))
        mean, cov = self.gaussianize_taylor(self.prior_ip, X_and_Z)

        # Sample from Q(u)
        u = self.generate_u_samples(num_samples) 
        # P(u) mean and covariance matrix
        Pu_mean = mean[n:]
        Pu_var = cov[n:, n:]
        

        # Pu_var shape (num_inducing, num_inducing, d)
        A = Pu_var.permute(2, 1, 0)
        A = torch.linalg.solve(A, cov[:n, n:].permute(2, 1, 0))
        A = A.permute(2, 1, 0)
        

        # Compute mean of P(f|u)
        Pfu_mean = mean[:n] + torch.einsum("nmd, smd -> snd", A, u - mean[n:])
        # Compute diagonal of P(f|u), variance does not depend on the value of u
        Pfu_var = torch.diagonal(cov[:n, :n]).T- torch.einsum(
            "nmd, mnd -> nd", A, cov[n:, :n]
        )
        # Replicate variance for every sample of Q(u)
        Pfu_var = torch.tile(Pfu_var.unsqueeze(0), (Pfu_mean.shape[0], 1, 1))

        return Pfu_mean, Pfu_var, Pu_mean, Pu_var

    def predict_y(self, predict_at, num_samples=None):

        Fmean, Fvar, _, _ = self.predict_f(predict_at, num_samples)
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

        F = self.prior_ip(X, num_samples)

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
            if param.requires_grad is False:
                continue
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


class FVI2(FVI):
    def gaussianize_samples(self, f):
        mean = torch.mean(f, dim=0)
        m = f - mean
        cov = torch.einsum("snd, smd ->nmd", m, m) / (m.shape[0] - 1)
        return mean, cov

    def predict_f(self, X, num_samples=None):

        # Batch size
        n = X.shape[0]
        # Concatenation of batch and inducing points
        X_and_Z = torch.concat([X, self.inducing_points], axis=0)

        # f([X, Z])

        # Compute P(f([X, Z]))
        F = self.prior_ip(X_and_Z, 500)

        mean, cov = self.gaussianize_samples(F)
        
        # Sample from Q(u)
        u = self.generate_u_samples(num_samples)

        # P(u) mean and covariance matrix
        Pu_mean = mean[n:]
        Pu_var = cov[n:, n:]

        # Pu_var shape (num_inducing, num_inducing, d)
        A = torch.linalg.solve(Pu_var.permute(2, 1, 0), cov[:n, n:].permute(2, 1, 0))
        A = A.permute(2, 1, 0)

        # Compute mean of P(f|u)
        Pfu_mean = mean[:n] + torch.einsum("abd, sbd -> sad", A, u - mean[n:])

        # Compute diagonal of P(f|u), variance does not depend on the value of u
        Pfu_var = torch.diagonal(cov[:n, :n]).T - torch.einsum(
            "abd, bad -> ad", A, cov[n:, :n]
        )

        # Replicate variance for every sample of Q(u)
        Pfu_var = torch.tile(Pfu_var.unsqueeze(0), (Pfu_mean.shape[0], 1, 1))

        return Pfu_mean, Pfu_var, Pu_mean, Pu_var, u

    # def KL(self, mu1, var1, mu2, var2):
    #     var1 = torch.diagonal(var1).T
    #     var2 = torch.diagonal(var2).T

    #     inv = torch.inverse(
    #         var2.transpose(0, -1) + 1e-6 * torch.eye(var2.shape[0])
    #     ).transpose(0, -1)

    #     KL = -mu2.shape[0]
    #     # Trace
    #     KL += torch.sum(var1 / var2, dim=[0])
    #     # Quadratic term
    #     KL += torch.sum((mu2 - mu1) ** 2 / var2, dim=0)

    #     KL += torch.log(torch.prod(var2, dim=0))
    #     KL -= torch.log(torch.prod(var1, dim=0))
    #     return 0.5 * torch.sum(KL)

    def predict_y(self, predict_at, num_samples=None):

        Fmean, Fvar, _, _, _ = self.predict_f(predict_at, num_samples)
        return self.likelihood.predict_mean_and_var(Fmean, Fvar)

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

        F_mean, F_var, Pu_mean, Pu_var, u = self.predict_f(X, self.num_samples)
        Qu_mean, Qu_var = self.gaussianize_samples(u)

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
        return -scale * ve + KL * 0


class FVI3(FVI):
    def __init__(
        self,
        prior_ip,
        variational_ip,
        Z,
        likelihood,
        num_data,
        fix_inducing,
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
                None,
                Z,
                likelihood,
                num_data,
                fix_inducing,
                num_samples,
                bb_alpha,
                y_mean,
                y_std,
                device,
                dtype,
                seed,
            )
        output_dim = self.prior_ip.output_dim
        # Regression Coefficients prior mean

        self.num_inducing = self.inducing_points.shape[0]
        self.output_dim = output_dim
        
        self.q_mu = torch.tensor(
            np.zeros((self.inducing_points.shape[0], output_dim)),
            dtype=self.dtype,
            device=self.device,
        )
        self.q_mu = torch.nn.Parameter(self.q_mu)
        
        q_sqrt = np.eye(self.inducing_points.shape[0])
        # Replicate it output_dim times
        # Shape (num_coeffs, num_coeffs, output_dim)
        q_sqrt = np.tile(q_sqrt[:, :, None], [1, 1, output_dim])
        # Create tensor with triangular representation.
        # Shape (output_dim, num_coeffs*(num_coeffs + 1)/2)
        li, lj = torch.tril_indices(self.inducing_points.shape[0],self.inducing_points.shape[0])
        triangular_q_sqrt = q_sqrt[li, lj]
        self.q_sqrt_tri = torch.tensor(
            triangular_q_sqrt,
            dtype=self.dtype,
            device=self.device,
        )
        self.q_sqrt_tri = torch.nn.Parameter(self.q_sqrt_tri)
        
        self.generator = GaussianSampler(2147483647, self.device)
        
    def generate_u_samples(self, num_samples):
        
        q_sqrt = (
            torch.zeros((self.inducing_points.shape[0], self.inducing_points.shape[0], self.prior_ip.output_dim))
            .to(self.dtype)
            .to(self.device)
        )
        li, lj = torch.tril_indices(self.inducing_points.shape[0], self.inducing_points.shape[0])
        q_sqrt[li, lj] = self.q_sqrt_tri
        
        z = self.generator((num_samples, self.inducing_points.shape[0], self.prior_ip.output_dim))
        
        samples = self.q_mu + torch.einsum("nsd, sfd -> nfd", z, q_sqrt)
        return samples
    
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

        F_mean, F_var, Pu_mean, Pu_var = self.predict_f(X, self.num_samples)


        ve = self.likelihood.variational_expectations(
            F_mean, F_var, y, alpha=self.bb_alpha
        )

        # Aggregate on data dimension
        ve = torch.sum(ve)
        # Scale loss term corresponding to minibatch size
        scale = self.num_data
        scale /= X.shape[0]

        # Compute KL term
        q_sqrt = (
            torch.zeros((self.inducing_points.shape[0], self.inducing_points.shape[0], self.prior_ip.output_dim))
            .to(self.dtype)
            .to(self.device)
        )
        li, lj = torch.tril_indices(self.inducing_points.shape[0], self.inducing_points.shape[0])
        q_sqrt[li, lj] = self.q_sqrt_tri
        KL = self.KL(self.q_mu, q_sqrt, Pu_mean, Pu_var)

        self.bb_alphas.append((-scale * ve).cpu().detach().numpy())
        self.KLs.append(KL.cpu().detach().numpy())
        # print(-scale * ve)
        # print(KL)
        return -scale * ve + KL
    
    def KL(self, Qu_mean, Qu_sqrt, Pu_mean, Pu_var):
        
        # d1 = torch.distributions.multivariate_normal.MultivariateNormal(
        #     loc = Qu_mean.transpose(0, -1),
        #     scale_tril = Qu_sqrt.permute(2, 0, 1)
        # )
        # d2 = torch.distributions.multivariate_normal.MultivariateNormal(
        #     loc = Pu_mean.transpose(0, -1)[0],
        #     covariance_matrix = Pu_var.permute(2, 0, 1)
        # )
        # KL = torch.sum(torch.distributions.kl.kl_divergence(d1, d2))
        # return KL

        I = 1e-6 * torch.eye(Pu_var.shape[0])

        L0 = Qu_sqrt.permute(2, 0, 1)
        L1 = torch.linalg.cholesky(Pu_var.permute(2, 0, 1) + I)

        M = torch.linalg.solve_triangular(L1, L0, upper=False)
        M = torch.diagonal(M, dim1=-2, dim2=-1)

        m = (Pu_mean - Qu_mean).transpose(0, -1).unsqueeze(-1)
        y = torch.linalg.solve_triangular(L1, m, upper=False).squeeze(-1)
        L1 = torch.diagonal(L1, dim1=-2, dim2=-1)
        L0 = torch.diagonal(L0, dim1=-2, dim2=-1)

        KL = -Pu_mean.shape[0]

        KL += torch.sum(M ** 2, dim=-1)
        KL += torch.sum(y ** 2, dim=-1)

        KL += 2 * torch.sum(torch.log(L1), dim=-1)
        KL -= 2 * torch.sum(torch.log(L0), dim=-1)

        KL = 0.5 * torch.sum(KL)
        return KL
        
    def predict_f(self, X, num_samples=None):

        # Batch size
        n = X.shape[0]
        # Concatenation of batch and inducing points
        X_and_Z = torch.concat([X, self.inducing_points], axis=0)
        # Compute P(f([X, Z]))
        mean, cov = self.gaussianize_taylor(self.prior_ip, X_and_Z)

        # Sample from Q(u)
        Pu_mean = mean[n:]
        Pu_var = cov[n:, n:]

        # Pu_var shape (num_inducing, num_inducing, d)
        A = Pu_var.permute(2, 1, 0) 
        A = torch.linalg.solve(A, cov[:n, n:].permute(2, 1, 0))
        A = A.permute(2, 1, 0)
        

        # Compute mean of P(f|u)
        Pfu_mean = mean[:n] + torch.einsum("nmd, md -> nd", A, self.q_mu - mean[n:])
        # Compute diagonal of P(f|u), variance does not depend on the value of u
        Pfu_var = torch.diagonal(cov[:n, :n]).T
        q_sqrt = (
            torch.zeros((self.num_inducing, self.num_inducing, self.output_dim))
            .to(self.dtype)
            .to(self.device)
        )
        li, lj = torch.tril_indices(self.num_inducing, self.num_inducing,)
        q_sqrt[li, lj] = self.q_sqrt_tri
        Q_var = torch.einsum("abd, cbd -> acd", q_sqrt, q_sqrt)

        Pfu_var = Pfu_var + torch.einsum(
            "nmd, mbd, nbd -> nd",
            A, 
            Q_var - Pu_var,
            A
        )

        # Replicate variance for every sample of Q(u)
        Pfu_var = torch.tile(Pfu_var.unsqueeze(0), (1, 1, 1))
        Pfu_mean = torch.tile(Pfu_mean.unsqueeze(0), (1, 1, 1))

        return Pfu_mean, Pfu_var, Pu_mean, Pu_var



class FVI4(FVI):
    def __init__(
        self,
        prior_ip,
        variational_ip,
        Z,
        likelihood,
        num_data,
        fix_inducing,
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
                None,
                Z,
                likelihood,
                num_data,
                fix_inducing,
                num_samples,
                bb_alpha,
                y_mean,
                y_std,
                device,
                dtype,
                seed,
            )
        output_dim = self.prior_ip.output_dim
        # Regression Coefficients prior mean

        self.num_inducing = self.inducing_points.shape[0]
        self.output_dim = output_dim
        
        self.q_mu = torch.tensor(
            np.zeros((self.inducing_points.shape[0], output_dim)),
            dtype=self.dtype,
            device=self.device,
        )
        self.q_mu = torch.nn.Parameter(self.q_mu)
        
        q_sqrt = np.eye(self.inducing_points.shape[0])
        # Replicate it output_dim times
        # Shape (num_coeffs, num_coeffs, output_dim)
        q_sqrt = np.tile(q_sqrt[:, :, None], [1, 1, output_dim])
        # Create tensor with triangular representation.
        # Shape (output_dim, num_coeffs*(num_coeffs + 1)/2)
        li, lj = torch.tril_indices(self.inducing_points.shape[0],self.inducing_points.shape[0])
        triangular_q_sqrt = q_sqrt[li, lj]
        self.q_sqrt_tri = torch.tensor(
            triangular_q_sqrt,
            dtype=self.dtype,
            device=self.device,
        )
        self.q_sqrt_tri = torch.nn.Parameter(self.q_sqrt_tri)
        
        self.generator = GaussianSampler(2147483647, self.device)

    def gaussianize_samples(self, f):
        mean = torch.mean(f, dim=0)
        m = f - mean
        cov = torch.einsum("snd, smd ->nmd", m, m) / (m.shape[0] - 1)
        return mean, cov


    def generate_u_samples(self, num_samples):
        
        q_sqrt = (
            torch.zeros((self.inducing_points.shape[0], self.inducing_points.shape[0], self.prior_ip.output_dim))
            .to(self.dtype)
            .to(self.device)
        )
        li, lj = torch.tril_indices(self.inducing_points.shape[0], self.inducing_points.shape[0])
        q_sqrt[li, lj] = self.q_sqrt_tri
        
        z = self.generator((num_samples, self.inducing_points.shape[0], self.prior_ip.output_dim))
        
        samples = self.q_mu + torch.einsum("nsd, sfd -> nfd", z, q_sqrt)
        return samples
    
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

        F_mean, F_var, Pu_mean, Pu_var = self.predict_f(X, self.num_samples)


        ve = self.likelihood.variational_expectations(
            F_mean, F_var, y, alpha=self.bb_alpha
        )

        # Aggregate on data dimension
        ve = torch.sum(ve)
        # Scale loss term corresponding to minibatch size
        scale = self.num_data
        scale /= X.shape[0]

        # Compute KL term
        q_sqrt = (
            torch.zeros((self.inducing_points.shape[0], self.inducing_points.shape[0], self.prior_ip.output_dim))
            .to(self.dtype)
            .to(self.device)
        )
        li, lj = torch.tril_indices(self.inducing_points.shape[0], self.inducing_points.shape[0])
        q_sqrt[li, lj] = self.q_sqrt_tri
        KL = self.KL(self.q_mu, q_sqrt, Pu_mean, Pu_var)

        self.bb_alphas.append((-scale * ve).cpu().detach().numpy())
        self.KLs.append(KL.cpu().detach().numpy())
        # print(-scale * ve)
        # print(KL)
        return -scale * ve + KL
    
    def KL(self, Qu_mean, Qu_sqrt, Pu_mean, Pu_var):
        
        # d1 = torch.distributions.multivariate_normal.MultivariateNormal(
        #     loc = Qu_mean.transpose(0, -1),
        #     scale_tril = Qu_sqrt.permute(2, 0, 1)
        # )
        # d2 = torch.distributions.multivariate_normal.MultivariateNormal(
        #     loc = Pu_mean.transpose(0, -1)[0],
        #     covariance_matrix = Pu_var.permute(2, 0, 1)
        # )
        # KL = torch.sum(torch.distributions.kl.kl_divergence(d1, d2))
        # return KL

        I = 1e-6 * torch.eye(Pu_var.shape[0])

        L0 = Qu_sqrt.permute(2, 0, 1)
        L1 = torch.linalg.cholesky(Pu_var.permute(2, 0, 1) + I)

        M = torch.linalg.solve_triangular(L1, L0, upper=False)
        M = torch.diagonal(M, dim1=-2, dim2=-1)

        m = (Pu_mean - Qu_mean).transpose(0, -1).unsqueeze(-1)
        y = torch.linalg.solve_triangular(L1, m, upper=False).squeeze(-1)
        L1 = torch.diagonal(L1, dim1=-2, dim2=-1)
        L0 = torch.diagonal(L0, dim1=-2, dim2=-1)

        KL = -Pu_mean.shape[0]

        KL += torch.sum(M ** 2, dim=-1)
        KL += torch.sum(y ** 2, dim=-1)

        KL += 2 * torch.sum(torch.log(L1), dim=-1)
        KL -= 2 * torch.sum(torch.log(L0), dim=-1)

        KL = 0.5 * torch.sum(KL)
        return KL
        
    def predict_f(self, X, num_samples=None):

        # Batch size
        n = X.shape[0]
        # Concatenation of batch and inducing points
        X_and_Z = torch.concat([X, self.inducing_points], axis=0)

        F = self.prior_ip(X_and_Z, 500)
        mean, cov = self.gaussianize_samples(F)

        # Sample from Q(u)
        Pu_mean = mean[n:]
        Pu_var = cov[n:, n:]

        # Pu_var shape (num_inducing, num_inducing, d)
        A = Pu_var.permute(2, 1, 0) 
        A = torch.linalg.solve(A, cov[:n, n:].permute(2, 1, 0))
        A = A.permute(2, 1, 0)
        

        # Compute mean of P(f|u)
        Pfu_mean = mean[:n] + torch.einsum("nmd, md -> nd", A, self.q_mu - mean[n:])
        # Compute diagonal of P(f|u), variance does not depend on the value of u
        Pfu_var = torch.diagonal(cov[:n, :n]).T
        q_sqrt = (
            torch.zeros((self.num_inducing, self.num_inducing, self.output_dim))
            .to(self.dtype)
            .to(self.device)
        )
        li, lj = torch.tril_indices(self.num_inducing, self.num_inducing,)
        q_sqrt[li, lj] = self.q_sqrt_tri
        Q_var = torch.einsum("abd, cbd -> acd", q_sqrt, q_sqrt)

        Pfu_var = Pfu_var + torch.einsum(
            "nmd, mbd, nbd -> nd",
            A, 
            Q_var - Pu_var,
            A
        )

        # Replicate variance for every sample of Q(u)
        Pfu_var = torch.tile(Pfu_var.unsqueeze(0), (1, 1, 1))
        Pfu_mean = torch.tile(Pfu_mean.unsqueeze(0), (1, 1, 1))


        
        return Pfu_mean, Pfu_var, Pu_mean, Pu_var



class SparseGP(FVI):
    def __init__(
        self,
        prior_ip,
        variational_ip,
        Z,
        likelihood,
        num_data,
        fix_inducing,
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
                None,
                Z,
                likelihood,
                num_data,
                fix_inducing,
                num_samples,
                bb_alpha,
                y_mean,
                y_std,
                device,
                dtype,
                seed,
            )
        output_dim = self.prior_ip.output_dim
        # Regression Coefficients prior mean

        self.kernel = self.prior_ip.GP_cov
        self.mean = self.prior_ip.GP_mean
        
        
        self.num_inducing = self.inducing_points.shape[0]
        self.output_dim = output_dim
        
        self.q_mu = torch.tensor(
            np.zeros((self.inducing_points.shape[0], output_dim)),
            dtype=self.dtype,
            device=self.device,
        )
        self.q_mu = torch.nn.Parameter(self.q_mu)
        
        q_sqrt = np.eye(self.inducing_points.shape[0])
        # Replicate it output_dim times
        # Shape (num_coeffs, num_coeffs, output_dim)
        q_sqrt = np.tile(q_sqrt[:, :, None], [1, 1, output_dim])
        # Create tensor with triangular representation.
        # Shape (output_dim, num_coeffs*(num_coeffs + 1)/2)
        li, lj = torch.tril_indices(self.inducing_points.shape[0],self.inducing_points.shape[0])
        triangular_q_sqrt = q_sqrt[li, lj]
        self.q_sqrt_tri = torch.tensor(
            triangular_q_sqrt,
            dtype=self.dtype,
            device=self.device,
        )
        self.q_sqrt_tri = torch.nn.Parameter(self.q_sqrt_tri)
        
        self.generator = GaussianSampler(2147483647, self.device)

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

        F_mean, F_var, Pu_mean, Pu_var = self.predict_f(X, self.num_samples)


        ve = self.likelihood.variational_expectations(
            F_mean, F_var, y, alpha=self.bb_alpha
        )

        # Aggregate on data dimension
        ve = torch.sum(ve)
        # Scale loss term corresponding to minibatch size
        scale = self.num_data
        scale /= X.shape[0]

        # Compute KL term
        q_sqrt = (
            torch.zeros((self.inducing_points.shape[0], self.inducing_points.shape[0], self.prior_ip.output_dim))
            .to(self.dtype)
            .to(self.device)
        )
        li, lj = torch.tril_indices(self.inducing_points.shape[0], self.inducing_points.shape[0])
        q_sqrt[li, lj] = self.q_sqrt_tri
        KL = self.KL(self.q_mu, q_sqrt, Pu_mean, Pu_var)

        self.bb_alphas.append((-scale * ve).cpu().detach().numpy())
        self.KLs.append(KL.cpu().detach().numpy())
        # print(-scale * ve)
        # print(KL)
        return -scale * ve + KL
    
    def KL(self, Qu_mean, Qu_sqrt, Pu_mean, Pu_var):

        I = 1e-6 * torch.eye(Pu_var.shape[0])

        L0 = Qu_sqrt.permute(2, 0, 1)
        L1 = torch.linalg.cholesky(Pu_var + I)

        M = torch.linalg.solve_triangular(L1, L0, upper=False)
        M = torch.diagonal(M, dim1=-2, dim2=-1)

        m = (Pu_mean - Qu_mean).transpose(0, -1).unsqueeze(-1)
        y = torch.linalg.solve_triangular(L1, m, upper=False).squeeze(-1)
        L1 = torch.diagonal(L1, dim1=-2, dim2=-1)
        L0 = torch.diagonal(L0, dim1=-2, dim2=-1)

        KL = -Pu_mean.shape[0]

        KL += torch.sum(M ** 2, dim=-1)
        KL += torch.sum(y ** 2, dim=-1)

        KL += 2 * torch.sum(torch.log(L1), dim=-1)
        KL -= 2 * torch.sum(torch.log(L0), dim=-1)

        KL = 0.5 * torch.sum(KL)
        return KL
        
    def predict_f(self, X, num_samples=None):
        
        Ku = (self.kernel(self.inducing_points) + 2e-5 * torch.eye(self.num_inducing)).unsqueeze(0)

        Kf = (self.kernel(X) + 2e-5 * torch.eye(X.shape[0])).unsqueeze(0)
        Kfu = (self.kernel(X, self.inducing_points)).unsqueeze(0)
    
        
        self.Lu = torch.linalg.cholesky(Ku +  1e-4 * torch.eye(self.num_inducing))

        A = torch.linalg.solve_triangular(self.Lu.permute(0,2,1), Kfu, upper=True, left= False)
        A = torch.linalg.solve_triangular(self.Lu, A, upper=False, left = False)

        mean_u = self.mean(self.inducing_points)
        mean = self.mean(X) + torch.einsum("dnm, md -> nd", A, self.q_mu - mean_u)

        # Shape (S, S, D)
        self.q_sqrt = (
            torch.zeros((self.num_inducing, self.num_inducing, self.output_dim))
            .to(self.dtype)
            .to(self.device)
        )
        li, lj = torch.tril_indices(self.num_inducing, self.num_inducing)
        self.q_sqrt[li, lj] = self.q_sqrt_tri
        
        SK = torch.einsum("nmd, bmd->dnb", self.q_sqrt, self.q_sqrt) - Ku
        
        B = torch.einsum("dmb, dnb -> dmn", SK, A)
        
        delta_cov = torch.sum(A * B.permute(0, 2, 1), -1)
        K = torch.diagonal(Kf, dim1=1, dim2=2) + delta_cov
        
        # Replicate variance for every sample of Q(u)
        K = K.T.unsqueeze(0)
        mean = mean.unsqueeze(0)

        return mean, K, mean_u, Ku


    def generate_u_samples(self, num_samples):
        
        q_sqrt = (
            torch.zeros((self.inducing_points.shape[0], self.inducing_points.shape[0], self.prior_ip.output_dim))
            .to(self.dtype)
            .to(self.device)
        )
        li, lj = torch.tril_indices(self.inducing_points.shape[0], self.inducing_points.shape[0])
        q_sqrt[li, lj] = self.q_sqrt_tri
        
        z = self.generator((num_samples, self.inducing_points.shape[0], self.prior_ip.output_dim))
        
        samples = self.q_mu + torch.einsum("nsd, sfd -> nfd", z, q_sqrt)
        return samples