from src.layers import VIPLayer
import numpy as np
import torch
from src.generative_models import GaussianSampler, BayesianNN


class LinearProjection:
    def __init__(self, matrix):
        """
        Encapsulates a linear projection defined by a Matrix

        Parameters
        ----------
        matrix : Torch tensor of shape (N, M)
                 Contains the linear projection
        """
        self.P = matrix

    def __call__(self, inputs):
        """
        Applies the linear transformation to the given input.
        """
        return torch.matmul(inputs, torch.tensor(self.P.T))


def init_layers(
    X,
    Y,
    inner_dims,
    regression_coeffs=20,
    structure=[10, 10],
    activation=torch.tanh,
    noise_sampler=None,
    trainable_parameters=True,
    trainable_prior=True,
    seed=0,
):
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

    Y : tf.tensor of shape (num_data, output_dim)
        Contains the input labels

    inner_dims : integer or list of integers
                 Indicates the number of VIP layers to use. If
                 an integer is used, as many layers as its value
                 are created, with output dimension output_dim.
                 If a list is given, layers are created so that
                 the dimension of the data matches these values.

                 For example, inner_dims = [10, 3] creates 3
                 layers; one that goes from data_dim features
                 to 10, another from 10 to 3, and lastly from
                 3 to output_dim.

    regression_coeffs : integer
                        Number of regression coefficients to use.

    structure : list of integers
                Specifies the hidden dimensions of the Bayesian
                Neural Networks in each VIP.

    activation : callable
                 Non-linear function to apply at each inner
                 dimension of the Bayesian Network.

    noise_sampler : NoiseSampler
                    Specifies the noise generationfunction

    trainable_prior : boolean
                      Determines whether the prior function parameters
                      are trainable or not.

    seed : int
           Random seed

    Returns
    -------
    """

    # Initialice noise sampler using the given seed.
    if noise_sampler is None:
        noise_sampler = GaussianSampler(seed)

    # Create VIP layers. If integer, replicate output dimension
    if isinstance(inner_dims, (int, np.integer)):
        dims = np.concatenate(
            ([X.shape[1]], np.ones(inner_dims, dtype=int) * Y.shape[1])
        )
    # Otherwise, append thedata dimensions to the array.
    else:
        dims = [X.shape[1]] + inner_dims + [Y.shape[1]]

    # Initialize layers array
    layers = []
    # We maintain a copy of X, where each projection is applied. That is,
    # if two data reductions are made, the matrix of the second is computed
    # using the projected (from the first projection) data.
    X_running = np.copy(X)
    for (i, (dim_in, dim_out)) in enumerate(zip(dims[:-1], dims[1:])):
        # Las layer has no transformation
        if i == len(dims) - 2:
            mf = None

        # No dimension change, identity matrix
        elif dim_in == dim_out:
            mf = LinearProjection(np.identity(n=dim_in))

        # Dimensionality reduction, PCA using svd decomposition
        elif dim_in > dim_out:
            _, _, V = np.linalg.svd(X_running, full_matrices=False)

            mf = LinearProjection(V[:dim_out, :])
            # Apply the projection to the running data,
            X_running = X_running @ V[:dim_out].T

        else:
            raise NotImplementedError(
                "Dimensionality augmentation is not" " handled currently."
            )

        # Create the Generation function, i.e, the Bayesian Neural Network
        bayesian_network = BayesianNN(
            noise_sampler=noise_sampler,
            input_dim=dim_in,
            structure=structure,
            activation=activation,
            output_dim=dim_out,
        )

        # Create layer
        layers.append(
            VIPLayer(
                bayesian_network,
                num_regression_coeffs=regression_coeffs,
                num_outputs=dim_out,
                input_dim=dim_in,
                mean_function=mf,
                trainable=trainable_parameters,
            )
        )

    return layers
