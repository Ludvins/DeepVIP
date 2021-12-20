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
from utils.pytorch_learning import fit, fit_with_metrics, score

args = manage_experiment_configuration()

torch.manual_seed(2147483647)

# CUDA for PyTorch
use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")
torch.backends.cudnn.benchmark = True
vars(args)["device"] = device

train_indexes, test_indexes = train_test_split(np.arange(len(args.dataset)),
                                               test_size=0.1,
                                               random_state=2147483647)

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
#scheduler = torch.optim.lr_scheduler.ExponentialLR(opt, gamma=0.9999)

# Perform training
train_hist, val_hist = fit_with_metrics(
    dvip,
    train_loader,
    opt,
    val_generator=val_loader,
    #scheduler=scheduler,
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
ax3 = fig.add_subplot(2, 2, 2)
ax4 = fig.add_subplot(2, 2, 4)

loss = df[["LOSS"]].to_numpy().flatten()
ax3.plot(loss, label="Training loss")
ax3.legend()
ax3.set_title("Loss evolution")
ax4.plot(np.arange(loss.shape[0]//2, loss.shape[0]), loss[loss.shape[0]//2:], label="Training loss")
ax4.legend()
ax4.set_title("Loss evolution in last half of epochs")


ax1.plot(df[["RMSE"]].to_numpy(), label="Training RMSE")
ax1.plot(df_val[["RMSE"]].to_numpy(), label="Validation RMSE")
ax1.legend()
ax1.set_title("RMSE evolution")
ax2.plot(df[["NLL"]].to_numpy(), label="Training NLL")
ax2.plot(df_val[["NLL"]].to_numpy(), label="Validation NLL")
ax2.legend()
ax2.set_title("NLL evolution")
plt.savefig("plots/boston_energy_0.1_lr_0.1_scheduler.png", format = "png")
# plt.show()
