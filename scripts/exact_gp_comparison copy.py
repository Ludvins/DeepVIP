import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader
import sys
from sklearn.model_selection import train_test_split

sys.path.append(".")

from src.dvip import DVIP_Base
from src.layers_init import init_layers
from src.likelihood import Gaussian
from utils.dataset import (
    Test_Dataset,
    Training_Dataset,
    Synthetic_Dataset,
)
from utils.plotting_utils import build_plot_name, plot_train_test
from utils.process_flags import manage_experiment_configuration
from utils.pytorch_learning import fit, predict, fit_with_metrics, predict_prior_samples
from scripts.filename import create_file_name

args = manage_experiment_configuration()

torch.manual_seed(args.seed)

device = "cpu"  # torch.device("cuda:0" if use_cuda else "cpu")
vars(args)["device"] = device


################## DATASET ###################

# Generate train/test partition using split number
train_dataset, train_test_dataset, test_dataset = args.dataset.get_split(
    args.test_size, args.seed + args.split
)

train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
train_test_loader = DataLoader(train_test_dataset, batch_size=args.batch_size)
val_loader = DataLoader(test_dataset, batch_size=args.batch_size)


from matplotlib import pyplot as plt

plt.figure(figsize=(18, 10))
# plt.scatter(
#    test_dataset.inputs, test_dataset.targets, label="Test points", s=2, alpha=0.8
# )
#plt.scatter(
#    train_dataset.inputs,
#    train_dataset.targets * train_dataset.targets_std + train_dataset.targets_mean,
#    label="Training points",
#    s=20.0,
#)


################ DVIP TRAINING ##############

# Get VIP layers
layers = init_layers(train_dataset.inputs, train_dataset.output_dim, **vars(args))


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

opt = torch.optim.Adam(dvip.parameters(), lr=args.lr)

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

# Define optimizer and compile model
opt = torch.optim.Adam(dvip.parameters(), lr=args.lr)


# Change MC samples for test
dvip.num_samples = args.num_samples_test
test_mean, test_sqrt = predict(dvip, val_loader, device=args.device)

x = torch.tensor(train_dataset.inputs)
print(x.shape)

samples = dvip.vip_layers[0].generative_function(x)
print(samples.shape)
test_prior_samples = predict_prior_samples(dvip, val_loader)[0]

sort = np.argsort(test_dataset.inputs.flatten())



x = test_dataset.inputs.flatten()[sort]
x0 = 500
x1 = 350


import matplotlib.gridspec as gridspec
gs = gridspec.GridSpec(2, 3, height_ratios = [2.5,1.5])
fig = plt.figure(figsize = (16, 9))
ax2 = fig.add_subplot(gs[1, 0:2]) # row 0, col 0
ax3 = fig.add_subplot(gs[1, 2]) # row 0, col 1
ax1 = fig.add_subplot(gs[0, :]) # row 1, span all columns


print(test_dataset.inputs.flatten()[sort])
for prior_sample in test_prior_samples[:50]:
    ax1.plot(
        test_dataset.inputs.flatten()[sort],
        prior_sample.flatten()[sort],
    )

ax1.axvline(x[x0], color = "darkmagenta")
ax1.axvline(x[x1], color = "darkorange")

for prior_sample in test_prior_samples[:50]:
    ax1.scatter(x[x1], prior_sample.flatten()[sort][x1], color = "darkorange")
    ax1.scatter(x[x0], prior_sample.flatten()[sort][x0], color = "darkmagenta")

fx0 = []
fx1 = []

for prior_sample in test_prior_samples:
    fx0.append(prior_sample.flatten()[sort][x0])
    fx1.append(prior_sample.flatten()[sort][x1])
    
fx0 = np.array(fx0)
fx1 = np.array(fx1)

from scipy.stats import kde


x = np.linspace(-4, 4, 100)
k = kde.gaussian_kde(fx0)
sort = np.argsort(fx0)
ax2.fill_between(x, 0, k(x), color = "darkmagenta", alpha  = .1)
k1 = kde.gaussian_kde(fx1)
sort1 = np.argsort(fx1)
ax2.fill_between(x, 0, k1(x), color = "darkorange", alpha  = .1)
ax2.plot(x, k(x), color = "darkmagenta")
ax2.plot(x, k1(x), color = "darkorange") 


ax3.hist2d(fx0, fx1, density = True, bins = 50)#, cmap = "YlOrRd")
import matplotlib
cmap = matplotlib.cm.get_cmap('viridis')
ax3.set_facecolor(cmap(0))
#ax3.imshow(zi)
#ax3.pcolormesh(xi, yi, zi.reshape(xi.shape), shading='auto')
#plt.tight_layout()
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
ax2.tick_params(axis='x', labelsize= 20)
ax2.tick_params(axis='y', labelsize= 20)
ax3.tick_params(axis='x', labelsize= 20)
ax3.tick_params(axis='y', labelsize= 20)

plt.savefig("plots/vip_prior_" + create_file_name(args) + ".pdf", dpi=1000)
plt.show()
