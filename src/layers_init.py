import numpy as np
import torch

from src.generative_functions import GP, BayesianNN
from src.layers import VIPLayer


class LinearProjection:
    def __init__(self, matrix, device):
        """
        Encapsulates a linear projection defined by a Matrix

        Parameters
        ----------
        matrix : Torch tensor of shape (N, M)
                 Contains the linear projection
        """
        self.P = torch.tensor(matrix, dtype=torch.float64, device=device)

    def __call__(self, inputs):
        """
        Applies the linear transformation to the given input.
        """
        return torch.einsum("...a, ab -> ...b", inputs, self.P)


def init_layers(X, output_dim, vip_layers, genf, regression_coeffs,
                bnn_structure, activation, seed, device, dtype,
                fix_prior_noise, dropout, **kwargs):
    """
    Creates the Variational Implicit Process layers using the given
    information. If the dimensionality is reducen between layers,
    these are created with a mean function that projects the data
    to their maximum variance projection (PCA).

    If several projections are made, the first is computed over the
    original data, and, the following are applied over the already
    projected data.

    Parameters
    ----------
    X : tf.tensor of shape (num_data, data_dim)
        Contains the input features.
    output_dim : int
                 Number of output dimensions of the model.
    vip_layers : integer or list of integers
                 Indicates the number of VIP layers to use. If
                 an integer is used, as many layers as its value
                 are created, with output dimension output_dim.
                 If a list is given, layers are created so that
                 the dimension of the data matches these values.

                 For example, inner_dims = [10, 3] creates 3
                 layers; one that goes from data_dim features
                 to 10, another from 10 to 3, and lastly from
                 3 to output_dim.
    genf : string
           Indicates the generation function to use, can be, BNN or GP.
    regression_coeffs : integer
                        Number of regression coefficients to use.
    bnn_structure : list of integers
                    Specifies the hidden dimensions of the Bayesian
                    Neural Networks in each VIP.
    activation : callable
                 Non-linear function to apply at each inner
                 dimension of the Bayesian Network.
    dtype : data-type
                The dtype of the layer's computations and weights.
    device : torch.device
                 The device in which the computations are made.
    seed : int
               integer to use as seed for randomness.
    """

    # Create VIP layers. If integer, replicate input dimension
    if len(vip_layers) == 1:
        vip_layers = [X.shape[1]] * (vip_layers[0] - 1)
    # Otherwise, append thedata dimensions to the array.
    else:
        if vip_layers[-1] != output_dim:
            raise RuntimeError(
                "Last vip layer does not correspond with data label")
    dims = [X.shape[1]] + vip_layers + [output_dim]

    # Initialize layers array
    layers = []
    # We maintain a copy of X, where each projection is applied. That is,
    # if two data reductions are made, the matrix of the second is computed
    # using the projected (from the first projection) data.
    X_running = np.copy(X)
    for (i, (dim_in, dim_out)) in enumerate(zip(dims[:-1], dims[1:])):

        # Last layer has no transformation
        if i == len(dims) - 2:
            mf = None

        # No dimension change, identity matrix
        elif dim_in == dim_out:
            mf = LinearProjection(np.identity(n=dim_in), device=device)

        # Dimensionality reduction, PCA using svd decomposition
        elif dim_in > dim_out:
            _, _, V = np.linalg.svd(X_running, full_matrices=False)

            mf = LinearProjection(V[:dim_out, :].T, device=device)
            # Apply the projection to the running data,
            X_running = X_running @ V[:dim_out].T

        else:
            raise NotImplementedError(
                "Dimensionality augmentation is not handled currently.")

        # Create the Generation function
        if genf == "BNN":
            f = BayesianNN(
                input_dim=dim_in,
                structure=bnn_structure,
                activation=activation,
                output_dim=dim_out,
                dropout=dropout,
                fix_random_noise=fix_prior_noise,
                device=device,
                seed=seed,
                dtype=dtype,
            )
        elif genf == "GP":
            f = GP(
                input_dim=dim_in,
                output_dim=dim_out,
                seed=seed,
                fix_random_noise=fix_prior_noise,
                dtype=dtype,
                device=device,
            )

        # Create layer
        layers.append(
            VIPLayer(
                f,
                num_regression_coeffs=regression_coeffs,
                input_dim=dim_in,
                output_dim=dim_out,
                mean_function=mf,
                seed=seed,
                dtype=dtype,
                device=device,
            ))

    return layers
