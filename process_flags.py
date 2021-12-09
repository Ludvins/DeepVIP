import argparse
from load_data import SPGP, synthetic, test, boston
import torch
import numpy as np

from src.dataset import Boston_Dataset


def manage_experiment_configuration(args=None):

    if args is None:
        # Get parser arguments
        parser = get_parser()
        args = parser.parse_args()

    FLAGS = vars(args)

    # Manage Dataset
    if args.dataset_name == "SPGP":
        args.dataset = Boston_Dataset()

    FLAGS["activation_str"] = args.activation
    # Manage Generative function
    if args.genf == "BNN":
        FLAGS["bnn_structure"] = args.bnn_structure

        if args.activation == "tanh":
            FLAGS["activation"] = torch.tanh
        elif args.activation == "relu":
            FLAGS["activation"] = torch.relu
        elif args.activation == "softplus":
            FLAGS["activation"] = torch.nn.Softplus()
        elif args.activation == "sigmoid":
            FLAGS["activation"] = torch.sigmoid
        elif args.activation == "cos":
            FLAGS["activation"] = torch.cos

    if args.dtype == "float64":
        FLAGS["dtype"] = torch.float64

    return args


def check_data(X_train, y_train, verbose=1):
    if X_train.shape[0] != y_train.shape[0]:
        print("Labels and features differ in the number of samples")
        return

    # Compute data information
    n_samples = X_train.shape[0]
    input_dim = X_train.shape[1]
    output_dim = y_train.shape[1]
    y_mean = np.mean(y_train, axis=0)
    y_std = np.std(y_train, axis=0)

    if verbose > 0:
        print("Number of samples: ", n_samples)
        print("Input dimension: ", input_dim)
        print("Label dimension: ", output_dim)
        print("Labels mean value: ", y_mean)
        print("Labels standard deviation: ", y_std)

    return n_samples, input_dim, output_dim, y_mean, y_std


def get_parser():
    """
    Defines and returns a parser for DeepVIP experiments.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--num_samples_train",
        type=int,
        default=1,
        help="Number of Monte Carlo samples of the posterior to "
        "use during training",
    )
    parser.add_argument(
        "--num_samples_test",
        type=int,
        default=200,
        help="Number of Monte Carlo samples of the posterior to "
        "use during inference",
    )
    parser.add_argument(
        "--genf",
        type=str,
        default="BNN",
        help=("Generative function or model to use. Bayesian Neural Network"
              " (BNN), Gaussian Process (GP) or Gaussian Process with "
              " Inducing Points (GPI)"),
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="SPGP",
        help="Dataset to use (SPGP, synthethic or boston)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=20000,
        help="Training epochs",
    )
    parser.add_argument(
        "--vip_layers",
        type=int,
        default=[1],
        nargs="+",
        help="Variational implicit layers structure",
    )
    parser.add_argument(
        "--bnn_structure",
        type=int,
        default=[10, 10],
        nargs="+",
        help="Prior Bayesian network inner layers.",
    )
    parser.add_argument(
        "--regression_coeffs",
        type=int,
        default=20,
        help="Number of regression coefficients to use",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1000,
        help="Number of regression coefficients to use",
    )
    parser.add_argument(
        "--activation",
        type=str,
        default="cos",
        help="Activation function to use in the Bayesian NN",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=0.001,
        help="Training learning rate",
    )
    parser.add_argument("--warmup", type=int, default=0)
    parser.add_argument("--no-fix_prior_noise",
                        dest="fix_prior_noise",
                        action='store_false')
    parser.set_defaults(fix_prior_noise=True)
    parser.add_argument("--freeze_prior",
                        dest="freeze_prior",
                        action='store_true')
    parser.set_defaults(freeze_prior=False)
    parser.add_argument("--freeze_posterior",
                        dest="freeze_posterior",
                        action='store_true')
    parser.set_defaults(freeze_posterior=False)
    parser.add_argument("--show", dest="show", action='store_true')
    parser.set_defaults(show=False)
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
    )
    parser.add_argument(
        "--verbose",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="float64",
    )
    parser.add_argument(
        "--name_flag",
        type=str,
        default="",
    )

    return parser
