import argparse


def get_parser():
    """
    Defines and returns a parser for DeepVIP experiments.
    """
    parser = argparse.ArgumentParser()
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
        "--eager",
        type=bool,
        default=False,
        help="Tensorflow Eager execution",
    )
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
