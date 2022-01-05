import numpy as np
import torch
import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

import sys

sys.path.append(".")

from src.likelihood import Gaussian
from utils.dataset import Synthetic_Dataset, Test_Dataset, Training_Dataset
from utils.metrics import Metrics
from utils.process_flags import manage_experiment_configuration
from utils.pytorch_learning import fit, predict, score
from src.generative_functions import GaussianSampler


class BayesLinear(torch.nn.Module):
    def __init__(
        self,
        input_dim,
        output_dim,
        w_mean,
        w_log_std,
        b_mean,
        b_log_std,
        device=None,
        seed=0,
        dtype=torch.float64,
    ):
        """
        Generates samples from a stochastic Bayesian Linear function
        f(x) = w^T x + b,   where w and b follow a Gaussian distribution,
        parameterized by their mean and log standard deviation.

        Parameters:
        -----------
        input_dim : int
                    Dimensionality of the input values `x`.
        output_dim : int
                     Dimensionality of the function output.
        w_mean : torch tensor of shape (input_dim, output_dim)
                 Initial value w's mean.
        w_log_std : torch tensor of shape (input_dim, output_dim)
                    Logarithm of the initial standard deviation of w.
        b_mean : torch tensor of shape (1, output_dim)
                 Initial value b's mean.
        b_log_std : torch tensor of shape (1, output_dim)
                    Logarithm of the initial standard deviation of b.
        device : torch.device
                 The device in which the computations are made.
        fix_random_noise : boolean
                           Wether to reset the Random Generator's seed in each
                           iteration.
        seed : int
               Initial seed for the random number generator.
        dtype : data-type
                The dtype of the layer's computations and weights.
        """
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim

        self.gaussian_sampler = GaussianSampler(seed, device)

        self.device = device
        # Create trainable parameters
        self.weight_mu = torch.nn.Parameter(w_mean)
        self.weight_log_sigma = torch.nn.Parameter(w_log_std)
        self.bias_mu = torch.nn.Parameter(b_mean)
        self.bias_log_sigma = torch.nn.Parameter(b_log_std)

    def forward(self, inputs):

        # Check the given input is valid
        if inputs.shape[-1] != self.input_dim:
            raise RuntimeError(
                "Input shape does not match stored data dimension"
            )

        z_w_shape = (self.input_dim, self.output_dim)
        z_b_shape = (1, self.output_dim)

        # Generate Gaussian values
        z_w = self.gaussian_sampler(z_w_shape)
        z_b = self.gaussian_sampler(z_b_shape)

        # Perform reparameterization trick
        w = self.weight_mu + z_w * torch.exp(self.weight_log_sigma)
        b = self.bias_mu + z_b * torch.exp(self.bias_log_sigma)

        return inputs @ w + b


class BayesianNN(torch.nn.Module):
    def __init__(
        self,
        structure,
        activation,
        input_dim=1,
        output_dim=1,
        dropout=0.1,
        seed=2147483647,
        device=None,
        dtype=torch.float64,
    ):
        """
        Defines a Bayesian Neural Network.

        Parameters:
        -----------
        structure : array-like
                    Contains the inner dimensions of the Bayesian Neural
                    network. For example, [10, 10] symbolizes a Bayesian
                    network with 2 inner layers of width 10.
        activation : function
                     Activation function to use between inner layers.
        input_dim : int
                    Dimensionality of the input values `x`.
        output_dim : int
                     Dimensionality of the function output.
        device : torch.device
                 The device in which the computations are made.
        fix_random_noise : boolean
                           Wether to reset the Random Generator's seed in each
                           iteration.
        seed : int
               Initial seed for the random number generator.
        dtype : data-type
                The dtype of the layer's computations and weights.

        """
        super().__init__()
        self.dtype = dtype
        self.structure = structure
        self.activation = activation
        self.generator = torch.Generator()
        self.generator.manual_seed(seed)
        if dropout > 0.0:
            self.dropout = torch.nn.Dropout(dropout)
        else:
            self.dropout = lambda x: x
        # Create an array symbolizing the dimensionality of the data at
        # each inner layer.
        dims = [input_dim] + structure + [output_dim]
        layers = []

        # Loop over the input and output dimension of each sub-layer.
        for _in, _out in zip(dims, dims[1:]):
            layers.append(
                BayesLinear(
                    _in,
                    _out,
                    w_mean=torch.normal(
                        mean=0.0,
                        std=1.00,
                        size=(_in, _out),
                        generator=self.generator,
                    ).to(device),
                    w_log_std=torch.log(
                        torch.abs(
                            torch.normal(
                                mean=0.0,
                                std=1.0,
                                size=(_in, _out),
                                generator=self.generator,
                            ).to(device)
                        )
                    ),
                    b_mean=torch.normal(
                        mean=0.0,
                        std=1.0,
                        size=[_out],
                        generator=self.generator,
                    ).to(device),
                    b_log_std=torch.log(
                        torch.abs(
                            torch.normal(
                                mean=0.0,
                                std=1.0,
                                size=[_out],
                                generator=self.generator,
                            ).to(device)
                        )
                    ),
                    device=device,
                    dtype=dtype,
                )
            )
        self.layers = torch.nn.ModuleList(layers)

    def forward(self, inputs):
        """Forward pass over each inner layer, activation is applied on every
        but the last level.
        """

        x = inputs

        for layer in self.layers[:-1]:
            x = self.activation(layer(x))
            x = self.dropout(x)
        x = self.layers[-1](x)
        print(x.shape)
        x = x.unsqueeze(0)
        return x

    def train_step(self, optimizer, X, y):

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
        loss_function = torch.nn.NLLLoss()
        predictions = self(X)
        print(predictions.shape)
        print(y.shape)
        loss = loss_function(predictions.squeeze(), y.squeeze())
        # Create backpropagation graph
        loss.backward()
        # Make optimization step
        optimizer.step()

        return loss


args = manage_experiment_configuration()

torch.manual_seed(2147483647)

# CUDA for PyTorch
use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")
torch.backends.cudnn.benchmark = True
vars(args)["device"] = device

n_splits = 20
results = pd.DataFrame(columns=Metrics().get_dict().keys())

for split in range(n_splits):
    train_indexes, test_indexes = train_test_split(
        np.arange(len(args.dataset)),
        test_size=0.1,
        random_state=2147483647 + split,
    )

    train_dataset = Training_Dataset(
        args.dataset.inputs[train_indexes],
        args.dataset.targets[train_indexes],
        verbose=False,
    )
    test_dataset = Test_Dataset(
        args.dataset.inputs[test_indexes],
        args.dataset.targets[test_indexes],
        train_dataset.inputs_mean,
        train_dataset.inputs_std,
    )
    # Initialize DataLoader
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size)

    BNN = BayesianNN(
        args.bnn_structure,
        args.activation,
        train_dataset.input_dim,
        train_dataset.output_dim,
        device=args.device,
    )
    opt = torch.optim.Adam(BNN.parameters(), lr=args.lr)

    fit(
        BNN,
        train_loader,
        opt,
        # scheduler=scheduler,
        epochs=args.epochs,
        device=args.device,
    )

    test_metrics = score(BNN, test_loader, device=args.device)

    results = results.append(test_metrics.get_dict(), ignore_index=True)

results.to_csv(
    path_or_buf="results/dataset={}_exactGP.csv".format(
        args.dataset_name,
        str(args.vip_layers[0]),
        str(args.dropout),
        args.lr,
        "-".join(str(i) for i in args.bnn_structure),
    ),
    encoding="utf-8",
)
