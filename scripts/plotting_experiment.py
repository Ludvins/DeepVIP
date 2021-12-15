import numpy as np
import torch
from torch.utils.data import DataLoader
import sys
sys.path.append(".")

from src.dvip import DVIP_Base
from src.layers_init import init_layers
from src.likelihood import Gaussian
from utils.dataset import DVIP_Dataset, Test_Dataset, Training_Dataset
from utils.plotting_utils import build_plot_name, plot_train_test
from utils.process_flags import manage_experiment_configuration
from utils.pytorch_learning import fit, predict, predict_prior_samples, score

args = manage_experiment_configuration()

# CUDA for PyTorch
use_cuda = torch.cuda.is_available()
device = "cpu"  # torch.device("cuda:0" if use_cuda else "cpu")
torch.backends.cudnn.benchmark = True
vars(args)["device"] = device

X_train, y_train, X_test, y_test = synthetic()

train_dataset = Training_Dataset(X_train, y_train)
test_dataset = Test_Dataset(X_test, y_test)

train_loader = DataLoader(train_dataset, batch_size=args.batch_size)
predict_loader = DataLoader(train_dataset, batch_size=args.batch_size)
test_loader = DataLoader(test_dataset, batch_size=len(test_dataset))

# Get VIP layers
layers = init_layers(
    train_dataset.inputs, train_dataset.output_dim, **vars(args)
)


# Instantiate Likelihood
ll = Gaussian()

# Create DVIP object
dvip = DVIP_Base(
    ll,
    layers,
    len(train_dataset),
    num_samples_train=args.num_samples_train,
    num_samples_eval=args.num_samples_test,
    y_mean=train_dataset.targets_mean,
    y_std=train_dataset.targets_std,
    device=args.device,
)

if args.freeze_prior:
    dvip.freeze_prior()
if args.freeze_posterior:
    dvip.freeze_posterior()

dvip.print_variables()

# Define optimizer and compile model
opt = torch.optim.Adam(dvip.parameters(), lr=args.lr)
# scheduler = torch.optim.lr_scheduler.ExponentialLR(opt, gamma=0.996)

# Perform training
train_metrics = fit(
    dvip, train_loader, opt, epochs=args.epochs, device=args.device
)
test_metrics = score(dvip, test_loader, device=args.device)
dvip.print_variables()
print("TEST RESULTS: ")
print("\t - NELBO: {}".format(test_metrics["LOSS"]))
print("\t - NLL: {}".format(test_metrics["NLL"]))
print("\t - RMSE: {}".format(test_metrics["RMSE"]))
print("\t - CRPS: {}".format(test_metrics["CRPS"]))

# Predict Train and Test
dvip.num_samples = args.num_samples_train
train_mean, train_var = predict(dvip, predict_loader, device=args.device)
train_prediction_mean, train_prediction_var = dvip.get_predictive_results(
    train_mean, train_var
)

# Change MC samples for test
dvip.num_samples = args.num_samples_test
test_mean, test_var = predict(dvip, test_loader, device=args.device)
test_prediction_mean, test_prediction_var = dvip.get_predictive_results(
    test_mean, test_var
)

dvip.num_samples = args.num_samples_train
train_prior_samples = predict_prior_samples(dvip, predict_loader)
test_prior_samples = predict_prior_samples(dvip, test_loader)

# Create plot title and path
fig_title, path = build_plot_name(dataset=args.dataset_name, **vars(args))

plot_train_test(
    train_mixture_means=train_mean,
    train_prediction_mean=train_prediction_mean,
    train_prediction_sqrt=np.sqrt(train_prediction_var),
    test_mixture_means=test_mean,
    test_prediction_mean=test_prediction_mean,
    test_prediction_sqrt=np.sqrt(test_prediction_var),
    X_train=X_train.flatten(),
    y_train=y_train.flatten(),
    X_test=X_test.flatten(),
    y_test=y_test.flatten() if y_test is not None else None,
    train_prior_samples=train_prior_samples[-1],
    test_prior_samples=test_prior_samples[-1],
    title=fig_title,
    path=path,
    show=args.show,
)
