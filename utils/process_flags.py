import argparse

import numpy as np
import torch

from src.generative_functions import BayesLinear, SimplerBayesLinear
from src.likelihood import BroadcastedLikelihood, Gaussian, MultiClass, Bernoulli
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

    # Manage Dataset
    args.dataset = get_dataset(args.dataset_name)

    FLAGS["metrics_type"] = args.dataset.type
    if args.dataset.type == "regression":
        args.likelihood = Gaussian(dtype=args.dtype, device=args.device)
        args.metrics = MetricsRegression
    elif args.dataset.type == "multiclass":
        mc = MultiClass(
            num_classes=args.dataset.classes, device=args.device, dtype=args.dtype
        )
        args.likelihood = BroadcastedLikelihood(mc)
        args.metrics = MetricsClassification

    elif args.dataset.type == "binaryclass":
        mc = Bernoulli(device=args.device, dtype=args.dtype
        )
        args.likelihood = BroadcastedLikelihood(mc)
        args.metrics = MetricsClassification

    FLAGS["activation_str"] = args.activation
    
    # Manage Generative function
    if args.bnn_structure == [0]:
        args.bnn_structure = []
    FLAGS["bnn_structure"] = args.bnn_structure

    FLAGS["bnn_layer_str"] = args.bnn_layer
    if args.bnn_layer == "BayesLinear":
        FLAGS["bnn_layer"] = BayesLinear
    elif args.bnn_layer == "SimplerBayesLinear":
        FLAGS["bnn_layer"] = SimplerBayesLinear
    else:
        raise ValueError("Invalid BNN layer type.")
    
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
    else:
        raise ValueError("Invalid BNN activation type.")


    if args.dtype == "float64":
        FLAGS["dtype"] = torch.float64

    len_train = args.dataset.len_train(args.test_size)
    args.batch_size = min(args.batch_size, len_train)
    if args.epochs is None:
        if args.iterations is None:
            raise ValueError("Either Epochs or Iterations must be selecetd.")
        args.epochs = int(
            np.round(args.iterations / np.ceil(len_train / args.batch_size))
        )

    return args


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
        default=100,
        help="Number of Monte Carlo samples of the posterior to "
        "use during inference",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
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
        "--bnn_layer",
        type=str,
        default="BayesLinear",
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
        type=lambda x: None if x == "None" else float(x),
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
        default=20,
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
        default=0.001,
        help="Training learning rate",
    )
    parser.add_argument(
        "--test_size",
        type=float,
        default=0.1,
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
        default=torch.float64,
    )
    parser.add_argument(
        "--split",
        default=None,
        type=int,
    )
    parser.add_argument("--name_flag", default="", type=str)
    
    parser.add_argument("--no_input_prop", dest="input_prop", action="store_false")
    parser.set_defaults(input_prop = True)
    parser.add_argument("--genf_full_output", dest="genf_full_output", action="store_true")
    parser.set_defaults(genf_full_output = False)

    return parser
