import argparse

import numpy as np
import torch
from .dataset import get_dataset


def manage_experiment_configuration(args=None):

    if args is None:
        # Get parser arguments
        parser = get_parser()
        args = parser.parse_args()

    FLAGS = vars(args)
    # Manage Dataset
    args.dataset = get_dataset(args.dataset_name)

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
    
    args.batch_size = min(args.batch_size, len(args.dataset))
    if args.epochs is None:
        if args.iterations is None:
            raise ValueError("Either Epochs or Iterations must be selecetd.")
        args.epochs = int(args.iterations / (len(args.dataset) / args.batch_size))

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
        help="Number of Monte Carlo samples of the posterior to " "use during training",
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
        help=(
            "Generative function or model to use. Bayesian Neural Network"
            " (BNN), Gaussian Process (GP) or Gaussian Process with "
            " Inducing Points (GPI)"
        ),
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        help="Dataset to use",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Training epochs",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=None,
        help="Training iterations",
    )
    parser.add_argument(
        "--vip_layers",
        type=int,
        default=[1],
        nargs="+",
        help="Variational implicit layers structure",
    )
    parser.add_argument(
        "--bb_alpha",
        type=float,
        default=0,
    )
    parser.add_argument(
        "--final_layer_mu",
        type=float,
        default=0,
    )
    parser.add_argument(
        "--final_layer_sqrt",
        type=float,
        default=1.0,
    )
    parser.add_argument(
        "--final_layer_noise",
        type=float,
        default=None,
    )
    parser.add_argument(
        "--inner_layers_mu",
        type=float,
        default=0.0,
    )
    parser.add_argument(
        "--inner_layers_sqrt",
        type=float,
        default=1e-5,
    )
    parser.add_argument(
        "--inner_layers_noise",
        type=float,
        default=-5,
    )
    parser.add_argument(
        "--bnn_inner_dim",
        type=int,
        default=10,
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
        default=50,
        help="Number of regression coefficients to use",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=100,
        help="Number of regression coefficients to use",
    )
    parser.add_argument(
        "--activation",
        type=str,
        default="tanh",
        help="Activation function to use in the Bayesian NN",
    )
    parser.add_argument(
        "--dropout",
        type=float,
        default=0.0,
        help="Dropout to use in the Bayesian NN",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=0.01,
        help="Training learning rate",
    )
    parser.add_argument("--warmup", type=int, default=0)
    parser.add_argument(
        "--no-fix_prior_noise", dest="fix_prior_noise", action="store_false"
    )
    parser.set_defaults(fix_prior_noise=True)

    parser.add_argument("--prior_kl", dest="prior_kl", action="store_true")
    parser.set_defaults(prior_kl=False)

    parser.add_argument(
        "--zero_mean_prior", dest="zero_mean_prior", action="store_true"
    )
    parser.set_defaults(zero_mean_prior=False)

    parser.add_argument("--freeze_prior", dest="freeze_prior", action="store_true")
    parser.set_defaults(freeze_prior=False)

    parser.add_argument(
        "--freeze_posterior", dest="freeze_posterior", action="store_true"
    )
    parser.set_defaults(freeze_posterior=False)

    parser.add_argument("--freeze_ll", dest="freeze_ll", action="store_true")
    parser.set_defaults(freeze_ll=False)

    parser.add_argument("--show", dest="show", action="store_true")
    parser.set_defaults(show=False)
    parser.add_argument(
        "--seed",
        type=int,
        default=2147483647,
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
        "--split",
        default=None,
        type=int,
    )
    parser.add_argument("--name_flag", default="", type=str)

    return parser
