from datetime import datetime
import numpy as np
import torch
import pandas as pd
from torch.utils.data import DataLoader
import sys
from time import process_time as timer

sys.path.append(".")

from src.dvip import DVIP_Base, TVIP
from src.layers_init import init_layers
from src.layers import TVIPLayer, TVIP3Layer
from src.likelihood import QuadratureGaussian
from utils.process_flags import manage_experiment_configuration
from utils.pytorch_learning import fit, score, predict
from scripts.filename import create_file_name
from src.generative_functions import *

args = manage_experiment_configuration()

args.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
torch.manual_seed(args.seed)

train_dataset, train_test_dataset, test_dataset = args.dataset.get_split(
    args.test_size, args.seed + args.split
)

# Get VIP layers
f = BayesianNN(
    num_samples=args.regression_coeffs,
    input_dim=train_dataset.input_dim,
    structure=args.bnn_structure,
    activation=args.activation,
    output_dim=train_dataset.output_dim,
    layer_model=args.bnn_layer,
    dropout=args.dropout,
    fix_random_noise=args.fix_prior_noise,
    zero_mean_prior=args.zero_mean_prior,
    device=args.device,
    seed=args.seed,
    dtype=args.dtype,
)

layer = TVIP3Layer(
    f,
    num_regression_coeffs=args.regression_coeffs,
    input_dim=train_dataset.input_dim,
    output_dim=train_dataset.output_dim,
    add_prior_regularization=args.prior_kl,
    mean_function=None,
    q_mu_initial_value=0,
    log_layer_noise=None,
    q_sqrt_initial_value=1,
    n_coupling=args.n_coupling,
    dtype=args.dtype,
    device=args.device,
)

# Initialize DataLoader
train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
train_test_loader = DataLoader(train_test_dataset, batch_size=args.batch_size)
test_loader = DataLoader(test_dataset, batch_size=args.batch_size)


# Create DVIP object
dvip = TVIP(
    likelihood=args.likelihood,
    layer=layer,
    num_data=len(train_dataset),
    num_samples=args.num_samples_train,
    bb_alpha=args.bb_alpha,
    y_mean=train_dataset.targets_mean,
    y_std=train_dataset.targets_std,
    dtype=args.dtype,
    device=args.device,
)
if args.freeze_prior:
    dvip.freeze_prior()
dvip.print_variables()

# Define optimizer and compile model
opt = torch.optim.Adam(dvip.parameters(), lr=args.lr)

# Set the number of training samples to generate
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

# import matplotlib.pyplot as plt

# a = np.arange(len(loss) // 3, len(loss))
# plt.plot(a, loss[len(loss) // 3 :])
# plt.show()


dvip.print_variables()
import matplotlib.pyplot as plt

plt.figure(figsize=(16, 6))
ax1 = plt.subplot(2, 3, 1)
ax2 = plt.subplot(2, 3, 2)
ax3 = plt.subplot(2, 3, 3)
ax4 = plt.subplot(2, 3, 4)
ax5 = plt.subplot(2, 3, 5)
ax6 = plt.subplot(2, 3, 6)

n = len(loss)
ax1.plot(np.arange(n // 5, n), loss[n // 5 :])
ax2.plot(np.arange(n // 5, n), dvip.bb_alphas[n // 5 :])
ax3.plot(np.arange(n // 5, n), dvip.KLs[n // 5 :])
ax4.plot(loss)
ax5.plot(dvip.bb_alphas)
ax6.plot(dvip.KLs)
ax4.set_yscale("symlog")
ax5.set_yscale("symlog")
ax6.set_yscale("symlog")
ax1.set_title("Loss")
ax2.set_title("Data Fitting Term")
ax3.set_title("Regularizer Term")
plt.savefig("plots/loss_" + create_file_name(args) + ".pdf", format="pdf")
# plt.show()
# plt.show()
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


d = {
    **vars(args),
    **{"time": end - start},
    **{k + "_train": v for k, v in train_metrics.items()},
    **test_metrics,
}

df = pd.DataFrame.from_dict(d, orient="index").transpose()
df.to_csv(
    path_or_buf="results/tvip_" + create_file_name(args) + ".csv",
    encoding="utf-8",
)
