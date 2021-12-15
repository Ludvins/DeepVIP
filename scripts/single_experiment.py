import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
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

torch.manual_seed(2020)

# CUDA for PyTorch
use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")
torch.backends.cudnn.benchmark = True
vars(args)["device"] = device

train_indexes, test_indexes = train_test_split(np.arange(len(args.dataset)),
                                               test_size=0.1,
                                               random_state=42)

train_dataset = Training_Dataset(args.dataset.inputs[train_indexes],
                                 args.dataset.targets[train_indexes])
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
# dvip.freeze_prior()

# Define optimizer and compile model
opt = torch.optim.Adam(dvip.parameters(), lr=args.lr)
# opt = SAM(dvip.parameters(), torch.optim.Adam, lr = args.lr)
# scheduler = torch.optim.lr_scheduler.ExponentialLR(opt, gamma=0.9995)

# Perform training
train_hist, val_hist = fit(
    dvip,
    train_loader,
    opt,
    val_generator=val_loader,
    # scheduler = scheduler,
    epochs=args.epochs,
    device=args.device,
)

dvip.num_samples = args.num_samples_test
test_metrics = score(dvip, val_loader, device=args.device)

print("TEST RESULTS: ")
print("\t - NELBO: {}".format(test_metrics["LOSS"]))
print("\t - NLL: {}".format(test_metrics["NLL"]))
print("\t - RMSE: {}".format(test_metrics["RMSE"]))
print("\t - CRPS: {}".format(test_metrics["CRPS"]))

df = pd.DataFrame.from_dict(train_hist)
df_val = pd.DataFrame.from_dict(val_hist)

fig = plt.figure(figsize=(20, 10))

ax1 = fig.add_subplot(2, 2, 1)
ax2 = fig.add_subplot(2, 2, 3)
ax3 = fig.add_subplot(1, 2, 2)

ax3.plot(df[["LOSS"]].to_numpy(), label="Training loss")
ax3.legend()
ax1.plot(df[["RMSE"]].to_numpy(), label="Training RMSE")
ax1.plot(df_val[["RMSE"]].to_numpy(), label="Validation RMSE")
ax1.legend()
ax2.plot(df[["NLL"]].to_numpy(), label="Training NLL")
ax2.plot(df_val[["NLL"]].to_numpy(), label="Validation NLL")
ax2.legend()
plt.show()
