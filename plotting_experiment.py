from src.dataset import DVIP_Dataset
from utils import plot_train_test, check_data, build_plot_name, get_parser, plot_prior_over_layers
from load_data import SPGP, synthetic

import torch
from torch.utils.data import DataLoader

from src.train import predict_prior_samples, train, predict

from src.likelihood import Gaussian
from src.dvip import DVIP_Base
from src.layers_init import init_layers
from src.generative_models import GaussianSampler

# Parse dataset
parser = get_parser()
args = parser.parse_args()

# Load data
if args.dataset == "SPGP":
    X_train, y_train, X_test, y_test = SPGP()
elif args.dataset == "synthetic":
    X_train, y_train, X_test, y_test = synthetic()

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

n_samples, input_dim, output_dim, y_mean, y_std = check_data(
    X_train, y_train, verbose)
batch_size = args.batch_size or n_samples

# CUDA for PyTorch
use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")
torch.backends.cudnn.benchmark = True

# Gaussian Likelihood
ll = Gaussian()

# Define the noise sampler
noise_sampler = GaussianSampler(seed)

# Get VIP layers
layers = init_layers(
    X_train,
    y_train,
    vip_layers,
    regression_coeffs,
    bnn_structure,
    activation=activation,
    noise_sampler=noise_sampler,
    trainable_parameters=True,
    trainable_prior=True,
    seed=seed,
)

train_dataset = DVIP_Dataset(X_train, y_train)
test_dataset = DVIP_Dataset(X_test, y_test)

train_loader = DataLoader(train_dataset, batch_size=batch_size)
test_loader = DataLoader(test_dataset, batch_size=batch_size)
train_predict = DataLoader(train_dataset, batch_size=len(train_dataset))
test_predict = DataLoader(test_dataset, batch_size=len(test_dataset))

# Create DVIP object
dvip = DVIP_Base(ll,
                 layers,
                 len(train_loader.dataset),
                 num_samples=1,
                 y_mean=train_dataset.targets_mean,
                 y_std=train_dataset.targets_std,
                 warmup_iterations=warmup)

for name, param in dvip.named_parameters():
    if param.requires_grad:
        print(name, param.data)

#prior_samples = predict_prior_samples(dvip, train_predict)
#plot_prior_over_layers(X_train, prior_samples)

# Define optimizer and compile model
opt = torch.optim.Adam(dvip.parameters(), lr=lr)

# Perform training
train(dvip, train_loader, opt, epochs=args.epochs)

for name, param in dvip.named_parameters():
    if param.requires_grad:
        print(name, param.data)

# Predict Train and Test

train_mean, train_var = predict(dvip, train_loader)
test_mean, test_var = predict(dvip, test_loader)
prior_samples_train = predict_prior_samples(dvip, train_predict)
prior_samples_test = predict_prior_samples(dvip, test_predict)

train_mean = train_mean.mean(1)
train_var = train_var.mean(1)
test_mean = test_mean.mean(1)
test_var = test_var.mean(1)

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
    prior_samples_train,
    prior_samples_test,
    title=fig_title,
    path=path,
)
