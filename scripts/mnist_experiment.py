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
from utils.metrics import MetricsRegression, MetricsClassification
from utils.process_flags import manage_experiment_configuration
from utils.pytorch_learning import fit, fit_with_metrics, score
from scripts.filename import create_file_name

args = manage_experiment_configuration()

torch.manual_seed(args.seed)



# Get VIP layers
layers = init_layers(train_dataset.inputs, args.dataset.output_dim, **vars(args))

train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle = True)
train_test_loader = DataLoader(train_test_dataset, batch_size=args.batch_size)
val_loader = DataLoader(test_dataset, batch_size=args.batch_size)

ll = Gaussian(device=args.device)

# Create DVIP object
dvip = DVIP_Base(
    args.likelihood,
    layers,
    len(train_dataset),
    bb_alpha=args.bb_alpha,
    num_samples=args.num_samples_train,
    y_mean=train_dataset.targets_mean,
    y_std=train_dataset.targets_std,
    dtype=args.dtype,
    device=args.device,
)

dvip.print_variables()

# Define optimizer and compile model
opt = torch.optim.Adam(dvip.parameters(), lr=args.lr)


# Perform training
train_hist, val_hist = fit_with_metrics(
    dvip,
    train_loader,
    opt,
    args.metrics,
    val_generator=val_loader,
    epochs=args.epochs,
    device=args.device,
)


dvip.print_variables()

dvip.num_samples = args.num_samples_test
test_metrics = score(dvip, val_loader, args.metrics, device=args.device)
train_metrics = score(dvip, train_test_loader, args.metrics, device=args.device)

test_metrics_names = list(test_metrics.keys())
num_metrics = len(test_metrics_names)

print("TEST RESULTS: ")
for k, v in test_metrics.items():
    print("\t - {}: {}".format(k, v))

df = pd.DataFrame.from_dict(train_hist)
df_val = pd.DataFrame.from_dict(val_hist)

fig = plt.figure(figsize=(20, 10))


ax3 = fig.add_subplot(2, 2, 2)
ax4 = fig.add_subplot(2, 2, 4)

loss = df[["LOSS"]].to_numpy().flatten()
ax3.plot(loss, label="Training loss")
ax3.legend()
ax3.set_title("Loss evolution")
ax4.plot(
    np.arange(loss.shape[0] // 5, loss.shape[0]),
    loss[loss.shape[0] // 5 :],
    label="Training loss",
)
ax4.legend()
ax4.set_title("Loss evolution in last half of epochs")

for i, m in enumerate(test_metrics_names[1:]):
    ax = fig.add_subplot(num_metrics - 1, 2, 2*i + 1)
    ax.plot(df[[m]].to_numpy(), label="Training {}".format(m))
    ax.plot(df_val[[m]].to_numpy(), label="Validation {}".format(m))
    ymin, ymax = ax.get_ylim()
    d = (ymax - ymin) / 10
    ax.vlines(
        np.argmin(df[[m]].to_numpy()),
        np.min(df[[m]].to_numpy()) - d,
        np.min(df[[m]].to_numpy()) + d,
        color="black",
        label="Minimum value",
    )
    ax.vlines(
        np.argmin(df_val[[m]].to_numpy()),
        np.min(df_val[[m]].to_numpy()) - d,
        np.min(df_val[[m]].to_numpy()) + d,
        color="black",
    )
    ax.legend()
    ax.set_title("{} evolution".format(m))



plt.savefig("plots/" + create_file_name(args) + ".png")
# open file for writing
f = open("plots/" + create_file_name(args) + ".txt", "w")

d = {
    **{k + "_train": v for k, v in train_metrics.items()},
    **test_metrics,
}
# write file
f.write(str(d))

# close file
f.close()

if args.show:
    plt.show()
