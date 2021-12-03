import argparse
from load_data import SPGP, synthetic, test
import torch
import numpy as np


def manage_experiment_configuration():

    # Get parser arguments
    parser = get_parser()
    args = parser.parse_args()

    FLAGS = vars(args)

    # Manage Dataset
    if args.dataset == "SPGP":
        FLAGS["X_train"], FLAGS["y_train"], FLAGS["X_test"], FLAGS[
            "y_test"] = SPGP()
    elif args.dataset == "synthetic":
        FLAGS["X_train"], FLAGS["y_train"], FLAGS["X_test"], FLAGS[
            "y_test"] = synthetic()
    elif args.dataset == "test":
        FLAGS["X_train"], FLAGS["y_train"], FLAGS["X_test"], FLAGS[
            "y_test"] = test()

    check_data(args)

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
    elif args.genf == "GPI":
        FLAGS["num_inducing"] = args.num_inducing
        FLAGS["use_kmeans"] = args.use_kmeans

    if len(args.vip_layers) == 1:
        FLAGS["vip_layers"] = args.vip_layers[0]
    else:
        FLAGS["vip_layers"] = args.vip_layers

    return args


def check_data(args):
    if args.X_train.shape[0] != args.y_train.shape[0]:
        print("Labels and features differ in the number of samples")
        return

    d = vars(args)
    # Compute data information
    d["n_samples"] = args.X_train.shape[0]
    d["input_dim"] = args.X_train.shape[1]
    d["output_dim"] = args.X_train.shape[1]
    d["y_mean"] = np.mean(args.y_train)
    d["y_std"] = np.std(args.y_train)

    if args.verbose > 0:
        print("Number of samples: ", args.n_samples)
        print("Input dimension: ", args.input_dim)
        print("Label dimension: ", args.output_dim)
        print("Labels mean value: ", args.y_mean)
        print("Labels standard deviation: ", args.y_std)


def get_parser():
    """
    Defines and returns a parser for DeepVIP experiments.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--num_samples_train",
        type=int,
        default=3,
        help="Number of Monte Carlo samples of the posterior to "
        "use during training")
    parser.add_argument(
        "--num_samples_test",
        type=int,
        default=10,
        help="Number of Monte Carlo samples of the posterior to "
        "use during inference")
    parser.add_argument(
        "--genf",
        type=str,
        default="BNN",
        help=("Generative function or model to use. Bayesian Neural Network"
              " (BNN), Gaussian Process (GP) or Gaussian Process with "
              " Inducing Points (GPI)"))
    parser.add_argument(
        "--dataset",
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
        default=128,
        help="Number of regression coefficients to use",
    )
    parser.add_argument(
        "--num_inducing",
        type=int,
        default=50,
        help="The number of inducing points to use when using GPI.")
    parser.add_argument(
        "--use_kmeans",
        type=bool,
        default=True,
        help="Wether to use KMeans initialization for the inducing points"
        "of the GP.")
    parser.add_argument(
        "--activation",
        type=str,
        default="tanh",
        help="Activation function to use in the Bayesian NN",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=0.001,
        help="Training learning rate",
    )
    parser.add_argument("--warmup", type=int, default=0)
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
        "--name_flag",
        type=str,
        default="",
    )

    return parser
