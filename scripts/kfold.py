from datetime import datetime
import numpy as np
import torch
import pandas as pd
from sklearn.model_selection import train_test_split
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

# Total splits and pandas dataframe to store splits results
n_splits = 20
results = pd.DataFrame(columns=Metrics().get_dict().keys())

for split in range(n_splits):
    print("\rCurrent fold: ", split, "/", n_splits, end="")

    # Generate train/test partition using split number 
    train_indexes, test_indexes = train_test_split(
        np.arange(len(args.dataset)),
        test_size=0.1,
        random_state=2147483647 + split)

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

    # Initialize DataLoader
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size)

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
    #scheduler = torch.optim.lr_scheduler.ExponentialLR(opt, gamma=0.9999)

    # Set the number of training samples to generate
    dvip.num_samples = args.num_samples_train
    # Train the model
    fit(
        dvip,
        train_loader,
        opt,
        #scheduler=scheduler,
        epochs=args.epochs,
        device=args.device)
    
    # Set the number of test samples to generate
    dvip.num_samples = args.num_samples_test

    # Test the model
    test_metrics = score(dvip, test_loader, device=args.device)

    # Store split test metrics
    results = results.append(test_metrics, ignore_index=True)

# Store results in csv file
results.to_csv(
    path_or_buf=
    "results/dataset={}_vip_layers={}_dropout={}_lr={}_structure={}.csv".
    format(args.dataset_name, str(args.vip_layers[0]), str(args.dropout),
           args.lr, "-".join(str(i) for i in args.bnn_structure)),
    encoding='utf-8')
