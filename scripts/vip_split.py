from datetime import datetime
import numpy as np
import torch
import pandas as pd
from torch.utils.data import DataLoader
import sys
from time import process_time as timer

sys.path.append(".")

from src.dvip import DVIP_Base
from src.layers_init import init_layers

from utils.process_flags import manage_experiment_configuration
from utils.pytorch_learning import fit, score
from scripts.filename import create_file_name

args = manage_experiment_configuration()

torch.manual_seed(args.seed)

train_dataset, train_test_dataset, test_dataset = args.dataset.get_split(
    args.test_size, args.seed + args.split
)

# Get VIP layers
layers = init_layers(train_dataset.inputs, args.dataset.output_dim, **vars(args))

# Initialize DataLoader
train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
train_test_loader = DataLoader(train_test_dataset, batch_size=args.batch_size)
test_loader = DataLoader(test_dataset, batch_size=args.batch_size)

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
if args.freeze_prior:
    dvip.freeze_prior()
# Define optimizer and compile model
opt = torch.optim.Adam(dvip.parameters(), lr=args.lr)

# Set the number of training samples to generate
dvip.num_samples = 1
# Train the model
start = timer()
loss = fit(
    dvip,
    train_loader,
    opt,
    use_tqdm=True,
    return_loss=True,
    iterations=args.iterations,
    device=args.device,
)
end = timer()

dvip.print_variables()
dvip.num_samples = 1

# Test the model
train_metrics = score(
    dvip, train_test_loader, args.metrics, use_tqdm=True, device=args.device
)
test_metrics = score(dvip, test_loader, args.metrics, use_tqdm=True, device=args.device)
print("TRAIN RESULTS: ")
for k, v in train_metrics.items():
    print("\t - {}: {}".format(k, v))

print("TEST RESULTS: ")
for k, v in test_metrics.items():
    print("\t - {}: {}".format(k, v))

# import matplotlib.pyplot as plt

# a = np.arange(len(loss) // 3, len(loss))
# plt.plot(a, loss[len(loss) // 3 :])
# plt.show()

d = {
    **vars(args),
    **{"time": end - start},
    **{k + "_train": v for k, v in train_metrics.items()},
    **test_metrics,
}

df = pd.DataFrame.from_dict(d, orient="index").transpose()
df.to_csv(
    path_or_buf="results/vip_" + create_file_name(args) + ".csv",
    encoding="utf-8",
)
