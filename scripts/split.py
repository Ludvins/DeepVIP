from datetime import datetime
import numpy as np
import torch
import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
import sys
import time


sys.path.append(".")

from src.dvip import DVIP_Base
from src.layers_init import init_layers
from src.likelihood import Gaussian
from utils.dataset import Test_Dataset, Training_Dataset
from utils.metrics import Metrics
from utils.process_flags import get_parser, manage_experiment_configuration
from utils.pytorch_learning import fit, score

args = manage_experiment_configuration()

torch.manual_seed(2147483647)

# CUDA for PyTorch
use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")
torch.backends.cudnn.benchmark = True
vars(args)["device"] = device

# Generate train/test partition using split number 
train_indexes, test_indexes = train_test_split(
        np.arange(len(args.dataset)),
        test_size=0.1,
        random_state=2147483647 + args.split)


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
d = {**vars(args), **test_metrics}

df = pd.DataFrame.from_dict(d, orient='index').transpose()
t = time.strftime('%d-%m-%Y_%H:%M:%S')
df.to_csv(path_or_buf = "results/test/{}.csv".format(t))