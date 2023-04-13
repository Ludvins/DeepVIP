import torch
from .utils import reparameterize
import numpy as np

class VaLLA(torch.nn.Module):
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
        net_forward,
        Z,
        prior_std,
        likelihood,
        num_data,
        output_dim,
        backend,
        track_inducing_locations = False,
        fix_inducing_locations = False,
        inducing_classes = None,
        alpha = 0.0,
        y_mean=0.0,
        y_std=1.0,
        device=None,
        dtype=torch.float64,
    ):
        super().__init__()
        # Store data information
        self.num_data = num_data
        self.output_dim = output_dim
        # Store Black-Box alpha value
        self.alpha = alpha

        # Store targets mean and std.
        self.y_mean = torch.tensor(y_mean, device=device)
        self.y_std = torch.tensor(y_std, device=device)
        
        self.backend = backend
        self.net = net_forward
        self.num_inducing = Z.shape[0]
        
        self.prior_std = torch.tensor(
            prior_std, 
            device = device,  
            dtype = dtype)
        
        self.inducing_locations = torch.tensor(Z, device = device, dtype = dtype)
        if not fix_inducing_locations:
            self.inducing_locations = torch.nn.Parameter(self.inducing_locations)
        
        if inducing_classes is None:
            self.inducing_class = torch.tensor(
                np.tile(
                    np.arange(self.output_dim), 
                    reps = np.ceil(Z.shape[0] / self.output_dim).astype(int)
                    )[:Z.shape[0]],
                device = device, dtype = torch.long
            )
        else:
            self.inducing_class = torch.tensor(
                inducing_classes,
                device = device, 
                dtype = torch.long)

        self.track_inducing_locations = track_inducing_locations
        if track_inducing_locations:
            self.inducing_history = [self.inducing_locations.clone().detach().cpu().numpy()]

        # Store likelihood and Variational Implicit layers
        self.likelihood = likelihood

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
        
        self.log_variance = torch.nn.Parameter(
            torch.tensor(-5 * np.ones(output_dim), dtype = self.dtype, device = self.device)
            )
        
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
            self.inducing_history += [self.inducing_locations.clone().detach().cpu().numpy()]

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

        mean_pred, var_pred = self.predict_mean_and_var(X)  # Forward pass

        # Temporarily change the num data variable so that the
        # scale of the likelihood is correctly computed on the
        # test dataset.
        num_data = self.num_data
        self.num_data = X.shape[0]
        # Compute the loss with scaled data
        loss = self.nelbo(X, (y - self.y_mean) / self.y_std)
        self.num_data = num_data

        return loss, mean_pred, var_pred

    def jacobian_features(self, X, dims = None):
        """
        Computes the Jacobian of the deep model w.r.t the parameters evaluated on
        the input.
        
        Parameters
        ----------
        X : torch tensor of shape (batch_size, data_dim)
            Contains the input features.
            
        Returns
        -------
        
        J : torch tensor of shape (barch_size, output_dim, n_parameters)
            Contains the Jacobians
        """
        if dims is None:
            Js = self.backend.jacobians(x = X)
        else:
            Js = self.backend.jacobians_on_outputs(x = X, outputs = dims)
            
        return Js * self.prior_std


    def forward_prior(self, X):
        # Compute feature vectors (Gradients) Shape (batch_size, output_dim, n_params)
        phi_x, F_mean = self.jacobian_features(X)
        

        # Compute full matrices
        # n is the batch_size and m the number of inducing locations
        # s is used for the number of parameters
        # a and b are used for the output dimension
        
        # Shape (output_dim, output_dim, batch_size)
        Kx_diag = torch.einsum("nas, nbs -> abn", phi_x, phi_x) 
        
        # Shape [batch_size, output_dim, output_dim]
        Fvar =  Kx_diag.permute(2, 0, 1) 
        Fvar = Fvar + torch.diag_embed(torch.exp(self.log_variance))
        return F_mean, Fvar
    
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
        F_mean = self.net(X)
        # Transform flattened cholesky decomposition parameter into matrix
        L = torch.eye(self.num_inducing, dtype = self.dtype, device = self.device)
        li, lj = torch.tril_indices(self.num_inducing, self.num_inducing)
        # Shape (num_inducing, num_inducing)
        L[li, lj] = self.L

        # Compute feature vectors (Gradients) Shape (batch_size, output_dim, n_params)
        phi_x  = self.jacobian_features(X)

        # Compute feature vector of inducing locations  Shape (n_inducing, n_params)
        phi_z = self.jacobian_features(self.inducing_locations, self.inducing_class)
        # Clean gradients of inducing locations
        self.inducing_locations.grad = None
        
        # Compute full matrices
        # n is the batch_size and m the number of inducing locations
        # s is used for the number of parameters
        # a and b are used for the output dimension
        
        # Shape (output_dim, output_dim, batch_size)
        Kx_diag = torch.einsum("nas, nbs -> abn", phi_x, phi_x) 
        
        # Shape (output_dim, batch_size, num_inducing)
        Kxz = torch.einsum("nas, ms -> anm", phi_x, phi_z)
        # Shape (num_inducing, num_inducing)
        self.Kz = torch.einsum("ns, ms -> nm", phi_z, phi_z) 
        
        # Compute auxiliar matrices
        # Shape [num_inducing, num_inducing]
        #H = I + L^T @ self.Kz @ L
        I = torch.eye(self.num_inducing, dtype = self.dtype, device = self.device)
        H = I + torch.einsum("mn, ml, lk -> nk", L, self.Kz, L)

        # Shape [num_inducing, num_inducing]
        #A = L @ H^{-1} @ L^T
        A = torch.einsum("nm, ml, kl -> nk", L, torch.inverse(H), L)

        # Compute predictive diagonal
        # Shape [output_dim, output_dim, batch_size, batch_size]
        #K2 = Kxz @ A @ Kxz^T
        K2 = torch.einsum("anm, ml, bkl -> abnk", Kxz, A, Kxz)

        # Shape [output_dim, output_dim, batch_size]
        diag = torch.diagonal(K2, dim1=-2, dim2= -1)

        # Shape [batch_size, output_dim, output_dim]
        Fvar =  (Kx_diag - diag).permute(2, 0, 1) 
        Fvar = Fvar + torch.diag_embed(torch.exp(self.log_variance))
        return F_mean, Fvar
    
    def compute_KL(self):
        """
        Computes the Kulback-Leibler divergence between the variational distribution
        and the prior.
        """
        # Shape (num_inducing, num_inducing)
        I = torch.eye(self.num_inducing, dtype = self.dtype, device = self.device)
        L = torch.eye(self.num_inducing, dtype = self.dtype, device = self.device)
        li, lj = torch.tril_indices(self.num_inducing, self.num_inducing)
        # Shape (num_inducing, num_inducing)
        L[li, lj] = self.L
    
        
        # Compute auxiliar matrices
        # Shape [num_inducing, num_inducing]
        H = I + L.T @ self.Kz @ L 

        # Shape [num_inducing, num_inducing]
        A = L @ torch.inverse(H) @ L.T

        log_det = torch.logdet(H)

        trace = torch.sum(torch.diagonal(self.Kz @ A))


        KL = 0.5 * log_det - 0.5 * trace
        
        return torch.sum(KL)
    

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

        bb_alpha = self.likelihood.variational_expectations(F_mean, 
                                                            F_var, 
                                                            y,
                                                            alpha=self.alpha)
        

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
        return self.likelihood.predict_mean_and_var(Fmu, F_var)
        

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

    def freeze_cholesky(self):
        self.L.requires_grad = False
    
    def freeze_inducing(self):
        self.inducing_locations.requires_grad = False

class VaLLASampling(VaLLA):
    def name(self):
        return "Subsampling VaLLA"

    def __init__(
        self,
        net_forward,
        Z,
        prior_std,
        likelihood,
        num_data,
        n_samples,
        output_dim,
        backend,
        track_inducing_locations = False,
        fix_inducing_locations = False,
        inducing_classes = None,
        alpha = 0.0,
        y_mean=0.0,
        y_std=1.0,
        device=None,
        dtype=torch.float64,
        seed = 2147483647
    ):
        super().__init__(net_forward, Z, prior_std, likelihood, num_data, output_dim, 
                         backend, track_inducing_locations, fix_inducing_locations,
                         inducing_classes,
                         alpha, y_mean, y_std, device, dtype) 
        
        self.n_samples = n_samples

        self.generator = torch.Generator(device = device)
        self.generator.manual_seed(seed)

    def predict(self, X, n_samples = None):
        
        if n_samples is None:
            n_samples = self.n_samples
            

        # Latent mean and covariance.
        # F_mean shape (batch_size, output_dim)
        # F_var shape (batch_size, output_dim, output_dim)
        F_mean, F_var = self(X)
        # Shape (batch_size, output_dim, output_dim)
        cholesky = torch.linalg.cholesky(F_var + 1e-5 * torch.eye(F_var.shape[-1]))
        
        # Standard Gaussian samples 
        # Shape (num_samples, batch_size, output_dim)
        z = torch.randn(size = (n_samples, F_mean.shape[0], F_mean.shape[1]), generator = self.generator).to(self.dtype)
        
        # Latent samples Shape (num_samples, batch_size, output_dim)
        F =  F_mean + torch.einsum("snd, nda -> sna", z, cholesky)

        return F
    
    
    def nelbo(self, X, y):
        # Latent samples Shape (num_samples, batch_size, output_dim)
        F = self.predict(X)
        
        # log density of the targets given the samples 
        # Shape (num_samples, batch_size)
        log_p = self.likelihood.logp(X, F, y)
        # Shape (batch_size)
        ell = torch.mean(log_p, 0)
        print(ell[:10])
        input()
        # Aggregate on data dimension
        ell = torch.sum(ell)

        # Scale loss term corresponding to minibatch size
        scale = self.num_data
        scale /= X.shape[0]

        # Compute KL term
        KL = self.compute_KL()

        
        self.ell_history.append((-scale * ell).detach().cpu().numpy())
        self.kl_history.append(KL.detach().cpu().numpy())
        return -scale * ell + KL
    

    
    def predict_mean_and_var(self, X):
        raise NotImplementedError


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

        F = self.predict(X) 

        # Temporarily change the num data variable so that the
        # scale of the likelihood is correctly computed on the
        # test dataset.
        num_data = self.num_data
        self.num_data = X.shape[0]
        # Compute the loss with scaled data
        loss = self.nelbo(X, (y - self.y_mean) / self.y_std)
        self.num_data = num_data

        return loss, F

class OptimalVaLLA(VaLLA):
    def name(self):
        return "SparseLA"

    def __init__(
        self,
        X_train,
        net_forward,
        Z,
        prior_std,
        likelihood,
        num_data,
        output_dim,
        backend,
        track_inducing_locations = False,
        fix_inducing_locations = False,
        inducing_classes = None,
        alpha = 0.0,
        y_mean=0.0,
        y_std=1.0,
        device=None,
        dtype=torch.float64,
    ):
        super().__init__()
        self.X_train = X_train
        # Store data information
        self.num_data = num_data
        self.output_dim = output_dim
        # Store Black-Box alpha value
        self.alpha = alpha
        

        # Store targets mean and std.
        self.y_mean = torch.tensor(y_mean, device=device)
        self.y_std = torch.tensor(y_std, device=device)
        self.backend = backend
        self.net = net_forward
        self.num_inducing = Z.shape[0]
        
        self.prior_std = torch.tensor(prior_std, 
            device = device,  
            dtype = dtype)
        

        self.inducing_locations = torch.tensor(Z, device = device, dtype = dtype)
        if not fix_inducing_locations:
            self.inducing_locations = torch.nn.Parameter(self.inducing_locations)
        
        if inducing_classes is None:
            self.inducing_class = torch.tensor(
                np.tile(
                    np.arange(self.output_dim), 
                    reps = np.ceil(Z.shape[0] / self.output_dim).astype(int)
                    )[:Z.shape[0]],
                device = device, dtype = torch.long
            )
        else:
            self.inducing_class = torch.tensor(
                inducing_classes,
                device = device, 
                dtype = torch.long)

        self.track_inducing_locations = track_inducing_locations
        if track_inducing_locations:
            self.inducing_history = [self.inducing_locations.clone().detach().cpu().numpy()]

        # Store likelihood and Variational Implicit layers
        self.likelihood = likelihood

        # Set device and data type (precision)
        self.device = device
        self.dtype = dtype
        
    def optimal_cholesky(self):
        """
        Computes the Cholesky decomposition of the optimal matrix A.
        """
        
        
        phi_x = self.jacobian_features(self.X_train)
        # Compute feature vector of inducing locations
        phi_z = self.jacobian_features(self.inducing_locations)
        # Clean gradients of inducing locations
        self.inducing_locations.grad = None
        self.Kz = torch.einsum("nas, mbs -> abnm", phi_z, phi_z) 
        Kz_inv = torch.inverse(self.Kz + 1e-3 * torch.eye(self.num_inducing))
        Kxz = torch.einsum("nas, mbs -> abnm", phi_x, phi_z)
        

        A = Kz_inv @ Kxz.transpose(-2, -1) @ Kxz @ Kz_inv * 1/np.exp(self.likelihood.log_variance)
        L = torch.linalg.cholesky(A + 1e-3 * torch.eye(self.num_inducing))
    
        return L
        
        
    def forward(self, X):
        # Compute mean 
        F_mean = self.net(X)
        
        # Transform flattened cholesky decomposition parameter into matrix
        I = torch.eye(self.num_inducing, dtype = self.dtype, device = self.device)
        triang = self.optimal_cholesky()

        # Compute feature vectors (Gradients)
        phi_x = self.jacobian_features(X)
        
        # Compute feature vector of inducing locations
        phi_z = self.jacobian_features(self.inducing_locations)
        # Clean gradients of inducing locations
        self.inducing_locations.grad = None

        # Compute full matrices
        # n is the batch_size and m the number of inducing locations
        # s is used for the number of parameters
        # a and b are used for the output dimension
        Kx_diag = torch.einsum("nas, nbs -> abn", phi_x, phi_x) 
        Kxz = torch.einsum("nas, mbs -> abnm", phi_x, phi_z)
        self.Kz = torch.einsum("nas, mbs -> abnm", phi_z, phi_z) 
        
        # Compute auxiliar matrices
        # Shape [C, C, num_inducing, num_inducing]
        H = I + triang.transpose(-2 ,-1) @ self.Kz @ triang 
        # Shape [C, C, num_inducing, num_inducing]
        A = triang @ torch.inverse(H) @ triang.transpose(-2, -1)
    
        # Compute predictive diagonal
        # Shape [C, C, num_inducing, num_inducing]
        K2 = Kxz @ A @ Kxz.transpose(-2,-1)

        # Shape [C, C, num_inducing]
        diag = torch.diagonal(K2, dim1=-2, dim2= -1)

        return F_mean, (Kx_diag - diag).permute(2, 0, 1)

    
    def compute_KL(self):
        # Shape (num_outputs, num_inducing, num_inducing)
        I = torch.tile(
            torch.eye(self.num_inducing).unsqueeze(0).unsqueeze(0), 
            [self.output_dim, self.output_dim, 1, 1]
        )
        triang =  self.optimal_cholesky()
        
        # Compute auxiliar matrices
        # Shape [C, C, num_inducing, num_inducing]
        H = I + triang.transpose(-2 ,-1) @ self.Kz @ triang 
        # Shape [C, C, num_inducing, num_inducing]
        A = triang @ torch.inverse(H) @ triang.transpose(-2, -1)

        log_det = torch.log(torch.det(H))
        trace = torch.sum(torch.diagonal(self.Kz @ A, 0, -2, -1), -1)
        
        KL = 0.5 * log_det - 0.5 * trace
        
        return torch.sum(KL)
    

                   
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

        self.prior_std = torch.tensor(
            prior_std, 
            device = device,  
            dtype = dtype)
        
        self.likelihood_hessian = likelihood_hessian
        self.likelihood = likelihood
        
        
        # Set device and data type (precision)
        self.device = device
        self.dtype = dtype
    
    def jacobian_features(self, X):
        Js = self.backend.jacobians(x = X)
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
        
        Lambda = Lambda.permute(0,2,1,3).flatten(start_dim = 0, end_dim = 1).flatten(start_dim = 1, end_dim = 2)
        Lambda_inv = torch.inverse(Lambda + 1e-7 * torch.eye(Lambda.shape[0]))
        Lambda_inv = Lambda_inv.unflatten(0, (output_dim, n_data)).unflatten(2, (output_dim, n_data))
        # Shape (output_dim, output_dim, num_data, num_data)
        K = Kx + Lambda_inv
        
        K = K.flatten(start_dim = 0, end_dim = 1).flatten(start_dim = 1, end_dim = 2)
        K_inv = torch.inverse(K + 1e-7 * torch.eye(K.shape[0]))
        K_inv = K_inv.unflatten(0, (output_dim, n_data)).unflatten(2, (output_dim, n_data))
        self.inv = K_inv.permute(0, 2, 1, 3)


    def forward(self, X):
        # Shape (batch_size, output_dim)
        mean = self.net(X)

        # Shape (bath_size, output_dim, num_parameters)
        Jz = self.jacobian_features(X)
        
        # Shape (output_dim, output_dim, batch_size, batch_size)
        Kzz = torch.einsum("nas, mbs -> abnm", Jz, Jz)
        
        # Shape (output_dim, output_dim, batch_size, num_data)
        Kzx = torch.einsum("nas, mbs -> abnm", Jz, self.Jx)
       
        K2 =  torch.einsum("abnm, bcml, dckl -> adnk", Kzx, self.inv, Kzx)

        # Shape (output_dim, output_dim, batch_size, batch_size)
        KLLA = Kzz - K2
        # Shape (output_dim, output_dim, batch_size)
        var = torch.diagonal(KLLA, dim1 = -2, dim2 = -1)
        
        # Permute variance to have shape ( num_data, output_dim, output_dim)
        return mean, var.permute(2, 0, 1)
    
    
    def predict_mean_and_var(self, X):
        Fmu, F_var = self(X)
        return self.likelihood.predict_mean_and_var(Fmu, F_var)

        
class ELLA(torch.nn.Module):
    def name(self):
        return "SparseLA"

    def __init__(
        self,
        net_forward,
        n_samples,
        n_eigh,
        prior_std,
        likelihood_hessian,
        likelihood,
        backend,
        seed,
        device=None,
        dtype=torch.float64,
    ):
        super().__init__()

        self.backend = backend
        self.net = net_forward
        
        self.M = n_samples
        self.K = n_eigh
        
        self.seed = seed

        self.prior_std = torch.tensor(
            prior_std, 
            device = device,  
            dtype = dtype)
        
        self.likelihood_hessian = likelihood_hessian
        
        self.likelihood = likelihood
        
        # Set device and data type (precision)
        self.device = device
        self.dtype = dtype
        
    
    def fit(self, X_train, y_train):
        
        rng = np.random.default_rng(self.seed)
        indexes = rng.choice(np.arange(X_train.shape[0]),
                             self.M, 
                             replace = False)
        self.Xz = X_train[indexes]
        
        phi_z  = self.backend.jacobians(self.Xz)
        c = torch.nn.functional.one_hot(y_train[indexes].to(torch.long).squeeze(), phi_z.shape[1]).unsqueeze(-1)
        phi_z = torch.sum(phi_z * c, 1)
        
        K = torch.einsum("ns, ms -> nm", phi_z, phi_z)

        L, V = torch.linalg.eigh(K)

        L = torch.abs(L[-self.K:]).flip(-1)
        V = V[:, -self.K:].flip(-1)

        self.v = torch.einsum("ms, mk -> sk", 
                              phi_z,
                              V/torch.sqrt(L).unsqueeze(0)
                              )

        Lambda = self.likelihood_hessian(X_train, y_train)
        Lambda = torch.diag_embed(Lambda)

        
        Jtrain, _ = self.backend.jacobians(x = X_train)
        phi_train = torch.einsum("mds, sk -> dmk", Jtrain, self.v)
        
        G = torch.einsum("amk, abmn, bng -> kg", phi_train, Lambda, phi_train)
        G = G + torch.eye(self.K)/(self.prior_std**2)
        
        self.G_inv = torch.inverse(G)

    def forward(self, X_test):

        Jz  = self.backend.jacobians(x = X_test)

        phi = torch.einsum("mds, sk -> dmk", Jz, self.v)

        K_test = torch.einsum("ank, kg, bmg -> abnm", phi, self.G_inv, phi)

        mean = self.net(X_test)
        var = torch.diagonal(K_test, dim1 = -2, dim2= -1)
        
        return mean, var.permute(2, 0, 1)
    
    def predict_mean_and_var(self, X):
        Fmu, F_var = self(X)
        
        return self.likelihood.predict_mean_and_var(Fmu, F_var)