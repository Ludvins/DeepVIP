from datetime import datetime
import torch
import pandas as pd
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader
import sys

sys.path.append(".")

from src.dvip import DVIP_Base
from src.layers_init import init_layers
from src.likelihood import Gaussian
from utils.dataset import Test_Dataset, Training_Dataset
from utils.metrics import Metrics
from utils.process_flags import manage_experiment_configuration
from utils.pytorch_learning import fit, score

args = manage_experiment_configuration()

torch.manual_seed(2147483647)

# CUDA for PyTorch
use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")
torch.backends.cudnn.benchmark = True
vars(args)["device"] = device

n_splits = 10
results = pd.DataFrame(columns = Metrics().get_dict().keys())
kfold = KFold(n_splits, shuffle=True, random_state=2147483647)

for train_indexes, test_indexes in kfold.split(args.dataset.inputs):
    train_dataset = Training_Dataset(
        args.dataset.inputs[train_indexes],
        args.dataset.targets[train_indexes],
        verbose=False,
    )
    test_dataset = Test_Dataset(args.dataset.inputs[test_indexes],
                                args.dataset.targets[test_indexes],
                                train_dataset.inputs_mean,
                                train_dataset.inputs_std)

    # Get VIP layers
    layers = init_layers(train_dataset.inputs, train_dataset.output_dim,
                         **vars(args))

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size)
    val_loader = DataLoader(test_dataset, batch_size=args.batch_size)

    # Instantiate Likelihood
    ll = Gaussian()

    # Create DVIP object
    dvip = DVIP_Base(
        ll,
        layers,
        len(train_dataset),
        num_samples=args.num_samples_train,
        y_mean=train_dataset.targets_mean,
        y_std=train_dataset.targets_std,
        dtype=args.dtype,
        device=args.device,
    )

    # Define optimizer and compile model
    opt = torch.optim.Adam(dvip.parameters(), lr=args.lr)

    # Perform training
    dvip.num_samples = args.num_samples_train
    train_hist = fit(
        dvip,
        train_loader,
        opt,
        # val_generator=val_loader,
        epochs=args.epochs,
        verbose=0,
        device=args.device)
    dvip.num_samples = args.num_samples_test
    test_metrics = score(dvip, val_loader, device=args.device)

    results = results.append(test_metrics, ignore_index = True)
    print("FOLD RESULTS: ")
    print("\t - NELBO: {}".format(test_metrics["LOSS"]))
    print("\t - NLL: {}".format(test_metrics["NLL"]))
    print("\t - RMSE: {}".format(test_metrics["RMSE"]))

results.to_csv("results/{}_{}_{}_{}.csv".format(args.dataset_name, str(args.dropout), args.lr, datetime.now()))
print("TEST RESULTS: ")
print(results.mean().to_string())
