from src.dataset import DVIP_Dataset
from utils import (
    plot_train_test,
    check_data,
    build_plot_name,
    get_parser,
    plot_prior_over_layers,
)
from load_data import SPGP, synthetic, test

import torch
from torch.utils.data import DataLoader

from src.train import predict_prior_samples, train, predict

from src.likelihood import Gaussian
from src.dvip import DVIP_Base
from src.layers_init import init_layers

# Parse dataset
parser = get_parser()
args = parser.parse_args()

# Load data
if args.dataset == "SPGP":
    X_train, y_train, X_test, y_test = SPGP()
elif args.dataset == "synthetic":
    X_train, y_train, X_test, y_test = synthetic()
elif args.dataset == "test":
    X_train, y_train, X_test, y_test = test()

# Get experiments variables
epochs = args.epochs
bnn_structure = args.bnn_structure
seed = args.seed
regression_coeffs = args.regression_coeffs
lr = args.lr
verbose = args.verbose
warmup = args.warmup

if len(args.vip_layers) == 1:
    vip_layers = args.vip_layers[0]

if args.activation == "tanh":
    activation = torch.tanh
elif args.activation == "relu":
    activation = torch.relu
elif args.activation == "softplus":
    activation = torch.nn.Softplus()
elif args.activation == "sigmoid":
    activation = torch.sigmoid

n_samples, input_dim, output_dim, y_mean, y_std = check_data(
    X_train, y_train, verbose)
batch_size = args.batch_size or n_samples

# CUDA for PyTorch
use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")
torch.backends.cudnn.benchmark = True

# Gaussian Likelihood
ll = Gaussian()

# Get VIP layers
layers = init_layers(X_train,
                     y_train,
                     vip_layers,
                     regression_coeffs,
                     bnn_structure,
                     activation=activation,
                     seed=seed,
                     device=device)

train_dataset = DVIP_Dataset(X_train, y_train)
test_dataset = DVIP_Dataset(X_test, y_test, normalize=False)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
predict_loader = DataLoader(train_dataset, batch_size=len(train_dataset))
test_loader = DataLoader(test_dataset, batch_size=len(test_dataset))

# Create DVIP object
dvip = DVIP_Base(
    ll,
    layers,
    len(train_dataset),
    num_samples=10,
    y_mean=train_dataset.targets_mean,
    y_std=train_dataset.targets_std,
)

# Define optimizer and compile model
opt = torch.optim.Adam(dvip.parameters(), lr=0.1)

# Perform training
train(dvip, train_loader, opt, epochs=args.epochs)

dvip.print_variables()

# Predict Train and Test
train_mean, train_var = predict(dvip, predict_loader)
test_mean, test_var = predict(dvip, test_loader)

train_prior_samples = predict_prior_samples(dvip, predict_loader)
test_prior_samples = predict_prior_samples(dvip, test_loader)

# Create plot title and path
fig_title, path = build_plot_name(
    vip_layers,
    bnn_structure,
    input_dim,
    output_dim,
    epochs,
    n_samples,
    args.dataset,
    args.name_flag,
)

plot_train_test(
    (train_mean, train_var),
    (test_mean, test_var),
    X_train,
    y_train,
    X_test,
    y_test,
    train_prior_samples,
    test_prior_samples,
    title=fig_title,
    path=path,
)
