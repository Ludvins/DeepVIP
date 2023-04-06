import argparse

import numpy as np
import torch

from utils.metrics import MetricsClassification, MetricsRegression
from .dataset import get_dataset


def manage_experiment_configuration(args=None):

    if args is None:
        # Get parser arguments
        parser = get_parser()
        args = parser.parse_args()

    FLAGS = vars(args)

    if args.device == "gpu":
        # CUDA for PyTorch
        use_cuda = torch.cuda.is_available()
        args.device = torch.device("cuda:0" if use_cuda else "cpu")
        torch.backends.cudnn.benchmark = True

    if args.dtype == "float64":
        FLAGS["dtype"] = torch.float64

    return args


def get_parser():
    """
    Defines and returns a parser for DeepVIP experiments.
    """
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
    )

    parser.add_argument(
        "--dataset_name",
        type=str,
        help=(
            "Dataset to use. Options: SPGP, boston, energy, concrete, naval,"
            " kin8nm, yatch, power, protein, winered, CO2, MNIST, Rectangles,"
            " Year, Airline, HIGGS, SUSY, taxi"
        ),
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
        "--MAP_iterations",
        type=int,
        help="Training iterations",
    )    

    parser.add_argument(
        "--bb_alpha",
        type=float,
        default=0,
        help="Value of alpha in alpha energy divergence.",
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=100,
        help="Batch size",
    )
    parser.add_argument(
        "--num_inducing",
        type=int,
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=0.001,
        help="Training learning rate",
    )

    parser.add_argument(
        "--MAP_lr",
        type=float,
        default=0.001,
        help="MAP Training learning rate",
    )
    
    parser.add_argument(
        "--test_size",
        type=float,
        default=0.1,
        help="Test proportion for datasets with non defined train test splits.",
    )
    
    parser.add_argument(
        "--fix_inducing",
        dest="fix_inducing",
        action="store_true",
    )
    parser.set_defaults(fix_inducing=False)


    parser.add_argument(
        "--freeze_posterior",
        dest="freeze_posterior",
        action="store_true",
        help="Freeze the posterior parameters, set them unlearnable",
    )
    parser.set_defaults(freeze_posterior=False)

    parser.add_argument(
        "--freeze_ll",
        dest="freeze_ll",
        action="store_true",
        help="Freeze the likelihood parameters, set them unlearnable",
    )
    parser.set_defaults(freeze_ll=False)

    parser.add_argument("--seed", type=int, default=2147483647, help="Random seed.")
    parser.add_argument(
        "--verbose", type=int, default=1, help="Set to 0 to disable messages."
    )
    parser.add_argument("--dtype", type=str, default=torch.float64, help="Data type")
    parser.add_argument("--split", default=None, type=int, help="Data split to use.")
    parser.add_argument("--name_flag", default="", type=str)

    return parser
