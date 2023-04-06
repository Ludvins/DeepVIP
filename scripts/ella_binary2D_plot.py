from datetime import datetime
import numpy as np
import torch
import pandas as pd
from torch.utils.data import DataLoader
import sys
from time import process_time as timer
from properscoring import crps_gaussian

from scipy.cluster.vq import kmeans2
sys.path.append(".")

from utils.process_flags import manage_experiment_configuration
from utils.pytorch_learning import fit_map, predict, forward
from scripts.filename import create_file_name
from src.generative_functions import *
from src.sparseLA import ELLA, GPLLA
from src.likelihood import Bernoulli
from src.backpack_interface import BackPackInterface
from utils.models import get_mlp
from utils.dataset import get_dataset, Test_Dataset
from laplace import Laplace
from mpl_toolkits.axes_grid1 import make_axes_locatable

args = manage_experiment_configuration()

args.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
torch.manual_seed(args.seed)


args.dataset = get_dataset(args.dataset_name)

train_dataset, full_dataset, test_dataset = args.dataset.get_split(
    args.test_size, args.seed + args.split
)

# Initialize DataLoader
train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
full_loader = DataLoader(full_dataset, batch_size=args.batch_size)
test_loader = DataLoader(test_dataset, batch_size=args.batch_size)




f = get_mlp(train_dataset.inputs.shape[1], train_dataset.targets.shape[1], 
            [50, 50], torch.nn.Tanh, torch.nn.Sigmoid,
            args.device, args.dtype)


# Define optimizer and compile model
opt = torch.optim.Adam(f.parameters(), lr=args.MAP_lr)
criterion = torch.nn.BCELoss()

# Set the number of training samples to generate
# Train the model
start = timer()

loss = fit_map(
    f,
    train_loader,
    opt,
    criterion = criterion,
    use_tqdm=True,
    return_loss=True,
    iterations=args.MAP_iterations,
    device=args.device,
)

print("MAP Loss: ", loss[-1])
end = timer()


ella = ELLA(f[:-1], 
            args.num_inducing,
            np.min([args.num_inducing, 20]),
            prior_std = 1.0,
            likelihood_hessian=lambda x,y:  (f(x)*(1-f(x))).unsqueeze(-1).permute(1, 2, 0),
            likelihood=Bernoulli(device=args.device, 
                        dtype = args.dtype), 
            backend = BackPackInterface(f, "classification"),
            seed = args.seed,
            device = args.device,
            dtype = args.dtype)


ella.fit(torch.tensor(train_dataset.inputs, device = args.device, dtype = args.dtype),
         torch.tensor(train_dataset.targets, device = args.device, dtype = args.dtype))


lla = GPLLA(f[:-1], 
            prior_std = 1.0,
            likelihood_hessian=lambda x,y: (f(x)*(1-f(x))).unsqueeze(-1).permute(1, 2, 0),
            likelihood=Bernoulli(device=args.device, 
                        dtype = args.dtype), 
            backend = BackPackInterface(f, "classification"),
            device = args.device,
            dtype = args.dtype)


lla.fit(torch.tensor(train_dataset.inputs, device = args.device, dtype = args.dtype),
        torch.tensor(train_dataset.targets, device = args.device, dtype = args.dtype))

def sigmoid(x):
    return 1/(1 + np.exp(-x))



import matplotlib.pyplot as plt
import matplotlib.cm as cm
plt.rcParams['pdf.fonttype'] = 42
fig, axis = plt.subplots(4, 2, figsize=(15, 15))

color_map = plt.get_cmap('coolwarm') 
axis[0][0].scatter(train_dataset.inputs[:, 0], train_dataset.inputs[:, 1], c = color_map(train_dataset.targets), alpha = 0.8, label = "Training Dataset")
axis[0][0].set_title("Training Dataset")
xlims = axis[0][0].get_xlim()
ylims = axis[0][0].get_ylim()



n_samples = 100
x_vals = np.linspace(xlims[0], xlims[1], n_samples)
y_vals = np.linspace(ylims[0], ylims[1], n_samples)
X, Y = np.meshgrid(x_vals, y_vals)
positions = np.vstack([X.ravel(), Y.ravel()]).T

grid_dataset = Test_Dataset(positions)
grid_loader = torch.utils.data.DataLoader(grid_dataset, batch_size = args.batch_size, shuffle = False)


mean, var = predict(ella, grid_loader)

mean = mean.reshape(n_samples, n_samples)
var = var.reshape(n_samples, n_samples)


cbarticks = np.arange(0.0,1.5,0.5)
cp = axis[1][1].contourf(X, Y, mean, cbarticks, cmap = color_map, vmin = 0, vmax = 1)


map_pred_grid = f(torch.tensor(positions, dtype = args.dtype, device=args.device)).squeeze().detach().cpu().numpy().reshape(n_samples, n_samples)
cbarticks = np.arange(0.0,1.5,0.5)
cp = axis[0][1].contourf(X, Y, map_pred_grid, cbarticks, cmap = color_map, vmin = 0, vmax = 1)




cbarticks = np.arange(0.0,1.05,0.05)
cp = axis[2][0].contourf(X, Y, mean, cbarticks, cmap = color_map, vmin = 0, vmax = 1)

divider = make_axes_locatable(axis[2][0])
cax = divider.append_axes('right', size='5%', pad=0.05)

fig.colorbar(cp, cax=cax, orientation='vertical',ticks=cbarticks)
axis[2][0].set_title(r"$\mathbb{E}[y|\mathcal{D}, \mathbf{x}]$")



cp = axis[2][1].contourf(X, Y, var, cmap = cm.gray)
divider = make_axes_locatable(axis[2][1])
cax = divider.append_axes('right', size='5%', pad=0.05)

fig.colorbar(cp, cax=cax, orientation='vertical')
axis[2][1].set_title(r"$\mathbb{V}[y|\mathcal{D}, \mathbf{x}]$")


map_pred = f[:-1](torch.tensor(full_dataset.inputs, device = args.device, dtype = args.dtype)).detach().cpu().numpy()
map_pred = (1. * (sigmoid(map_pred) >= 0.5))



axis[0][1].scatter(full_dataset.inputs[:, 0], full_dataset.inputs[:, 1], c= color_map(map_pred),  alpha = 0.8, label = "Predictions")
axis[0][1].set_title("MAP predictions")



mean, _ = predict(ella, full_loader)


pred = (1. * (mean >= 0.5))
axis[1][1].scatter(full_dataset.inputs[:, 0], full_dataset.inputs[:, 1], c= color_map(pred),  alpha = 0.8, label = "Predictions")
axis[1][1].set_title("LLA predictions")


colors = cm.rainbow(np.linspace(0, 1, ella.Xz.shape[0]))
axis[1][0].scatter(ella.Xz[:, 0], ella.Xz[:, 1], c = colors,  alpha = 0.8, label = "Nystrom Locations")
axis[1][0].set_xlim(xlims[0], xlims[1])
axis[1][0].set_ylim(ylims[0], ylims[1])
axis[1][0].set_title("Nystrom locations")



axis[2][0].scatter(train_dataset.inputs[:, 0], train_dataset.inputs[:, 1], c = color_map(train_dataset.targets), alpha = 0.2)
axis[2][1].scatter(train_dataset.inputs[:, 0], train_dataset.inputs[:, 1], c = color_map(train_dataset.targets), alpha = 0.2)




_, ella_var = forward(ella, grid_loader)
ella_std = np.sqrt(ella_var.reshape(n_samples, n_samples))

_, lla_var = forward(lla, grid_loader)
lla_std = np.sqrt(lla_var.reshape(n_samples, n_samples))


levels = np.linspace(0,12,13)
cp = axis[3][0].contourf(X, Y, ella_std, levels = levels, cmap = cm.gray)
divider = make_axes_locatable(axis[3][0])
cax = divider.append_axes('right', size='5%', pad=0.05)

fig.colorbar(cp, cax=cax, orientation='vertical')
axis[3][0].set_title(r"ELLA Latent std")



cp = axis[3][1].contourf(X, Y, lla_std, levels = levels, cmap = cm.gray)
divider = make_axes_locatable(axis[3][1])
cax = divider.append_axes('right', size='5%', pad=0.05)

fig.colorbar(cp, cax=cax, orientation='vertical')
axis[3][1].set_title(r"LLA Latent std")


plt.savefig("ELLA_{}=M={}_K={}.pdf".format(args.dataset_name, args.num_inducing, ella.K), format="pdf")

plt.show()
