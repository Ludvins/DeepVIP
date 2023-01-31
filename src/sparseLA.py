import torch
from .utils import reparameterize
import numpy as np

from backpack import backpack, extend, memory_cleanup
from backpack.extensions import  BatchGrad
from backpack.context import CTX

def _cleanup(module):
        for child in module.children():
            _cleanup(child)

        setattr(module, "_backpack_extend", False)
        memory_cleanup(module)
        
        
class SparseLA(torch.nn.Module):
    def name(self):
        return "SparseLA"

    def __init__(
        self,
        net_forward,
        Z,
        prior_variance_init,
        likelihood,
        num_data,
        output_dim,
        backend,
        track_inducing_locations = False,
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
        
        self.prior_log_variance = torch.tensor(
            np.math.log(prior_variance_init), 
            device = device,  
            dtype = dtype)
        #self.prior_log_variance = torch.nn.Parameter(self.prior_log_variance)
        
        self.inducing_locations = torch.tensor(Z, device = device, dtype = dtype)
        self.inducing_locations = torch.nn.Parameter(self.inducing_locations)

        self.track_inducing_locations = track_inducing_locations
        if track_inducing_locations:
            self.inducing_history = [self.inducing_locations.clone().detach().cpu().numpy()]

        # Store likelihood and Variational Implicit layers
        self.likelihood = likelihood

        # Set device and data type (precision)
        self.device = device
        self.dtype = dtype
        
        I = np.eye(self.num_inducing) * 0.00001
        I = np.tile(I[None, :, :], [self.output_dim, 1, 1])
        
        
        li, lj = torch.tril_indices(self.num_inducing, self.num_inducing)
        triangular_q_sqrt = I[:, li, lj]
        self.L = torch.tensor(
            triangular_q_sqrt,
            dtype=self.dtype,
            device=self.device,
        )

        self.L = torch.nn.Parameter(self.L)
        

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


    def jacobian_features(self, X):
        Js, _ = self.backend.jacobians(x = X)

        return Js * torch.sqrt(torch.exp(self.prior_log_variance),)

    
    def forward(self, X):
        # Compute mean 
        F_mean = self.net(X).T
        
        # Transform flattened cholesky decomposition parameter into matrix
        I = torch.eye(self.num_inducing, dtype = self.dtype, device = self.device)
        triang = torch.tile(I.unsqueeze(0), [self.output_dim, 1, 1])
        li, lj = torch.tril_indices(self.num_inducing, self.num_inducing)
        triang[:, li, lj] = self.L

        # Compute feature vectors (Gradients)
        phi_x = self.jacobian_features(X)
        
        phi_z = self.jacobian_features(self.inducing_locations)
        self.inducing_locations.grad = None

        # Compute full matrices
        Kx_diag = torch.einsum("nds, nds -> dn", phi_x, phi_x) 
        Kxz = torch.einsum("nds, mds -> dnm", phi_x, phi_z)
        self.Kz = torch.einsum("nds, mds -> dnm", phi_z, phi_z) 
        
        # Compute auxiliar matrices
        H = I + triang.transpose(1 ,2) @ self.Kz @ triang 
        A = triang @ torch.inverse(H) @ triang.transpose(1, 2)
    
        
        K2 = Kxz @ A @ Kxz.transpose(1,2)
        diag = torch.diagonal(K2, dim1=1, dim2= 2)
        return F_mean.T, torch.sqrt(Kx_diag - diag).T
    
    def compute_KL(self):
        # Shape (num_outputs, num_inducing, num_inducing)
        I = torch.tile(torch.eye(self.num_inducing).unsqueeze(0), [self.output_dim, 1, 1])
        triang = torch.eye(self.num_inducing, dtype = self.dtype, device = self.device)
        triang = torch.tile(triang.unsqueeze(0), [self.output_dim, 1, 1])
        li, lj = torch.tril_indices(self.num_inducing, self.num_inducing)
        triang[:, li, lj] = self.L
    
        
        
        H = I + triang.transpose(1 ,2) @ self.Kz @ triang
        A = triang @ torch.inverse(H) @ triang.transpose(1, 2)

        
        log_det = torch.log(torch.det(H))
        trace = torch.sum(torch.diagonal(self.Kz @ A, 0, 1, 2), -1)
        
        KL = 0.5 * log_det - 0.5 * trace
        
        return torch.sum(KL)
    

    def nelbo(self, X, y):
        # Compute loss
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

        return -scale * bb_alpha + KL


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
