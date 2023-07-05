from datetime import datetime
import numpy as np
import torch
import pandas as pd
from torch.utils.data import DataLoader
import sys
from time import process_time as timer
from properscoring import crps_gaussian

import matplotlib.pyplot as plt
import matplotlib.cm as cm
from scipy.cluster.vq import kmeans2
sys.path.append(".")
from mpl_toolkits.axes_grid1 import make_axes_locatable

from utils.process_flags import manage_experiment_configuration
from utils.pytorch_learning import fit_map_crossentropy, fit, predict, forward, score
from scripts.filename import create_file_name
from src.generative_functions import *
from src.sparseLA import VaLLASampling, GPLLA
from src.likelihood import MultiClass
from src.backpack_interface import BackPackInterface
from utils.models import get_mlp
from utils.dataset import get_dataset, Test_Dataset
from laplace import Laplace
args = manage_experiment_configuration()

args.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
torch.manual_seed(args.seed)


args.dataset = get_dataset(args.dataset_name)

train_dataset, train_test_dataset, test_dataset = args.dataset.get_split(
    args.test_size, args.seed + args.split
)

# Initialize DataLoader
train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
train_test_loader = DataLoader(train_test_dataset, batch_size=args.batch_size)
test_loader = DataLoader(test_dataset, batch_size=args.batch_size)


def s():
    return torch.nn.Softmax(dim = -1)

f = get_mlp(train_dataset.inputs.shape[1], args.dataset.output_dim, 
            [50, 50], torch.nn.Tanh,
            device = args.device, dtype =args.dtype)

# Define optimizer and compile model
opt = torch.optim.Adam(f.parameters(), lr=args.MAP_lr)
criterion = torch.nn.CrossEntropyLoss()

try:
    f.load_state_dict(torch.load("weights/multiclass_weights"))
except:
    # Set the number of training samples to generate
    # Train the model
    start = timer()

    loss = fit_map_crossentropy(
        f,
        train_loader,
        opt,
        criterion = criterion,
        use_tqdm=True,
        return_loss=True,
        iterations=args.MAP_iterations,
        device=args.device,
        dtype = args.dtype
    )

    print("MAP Loss: ", loss[-1])
    end = timer()
    torch.save(f.state_dict(), "weights/multiclass_weights")



def hessian(x, y):
    #oh = torch.nn.functional.one_hot(y.long().flatten(), args.dataset.classes).type(args.dtype)
    out = torch.nn.Softmax(dim = -1)(f(x))
    a = torch.einsum("na, nb -> abn", out, out)
    b = torch.diag_embed(out).permute(1,2,0)
    #b = torch.sum(out * oh, -1)
    return - a + b


lla = GPLLA(f, 
            prior_std = args.prior_std,
            likelihood_hessian=lambda x,y: hessian(x, y),
            likelihood=MultiClass(num_classes = args.dataset.classes,
                          device=args.device, 
                        dtype = args.dtype), 
            backend = BackPackInterface(f, f.output_size),
            device = args.device,
            dtype = args.dtype)


lla.fit(torch.tensor(train_dataset.inputs, device = args.device, dtype = args.dtype),
        torch.tensor(train_dataset.targets, device = args.device, dtype = args.dtype))





plt.rcParams['pdf.fonttype'] = 42
fig, axis = plt.subplots(5, 3, figsize=(15, 20))

color_map = plt.get_cmap('tab10') 
axis[0][0].scatter(train_dataset.inputs[:, 0], train_dataset.inputs[:, 1], c = color_map(train_dataset.targets.astype(np.int32)), alpha = 0.8, label = "Training Dataset")
axis[0][0].set_title("Training Dataset")
xlims = axis[0][0].get_xlim()
ylims = axis[0][0].get_ylim()



n_samples = 50
x_vals = np.linspace(-3, 3, n_samples)
y_vals = np.linspace(-3, 3, n_samples)
X, Y = np.meshgrid(x_vals, y_vals)
positions = np.vstack([X.ravel(), Y.ravel()]).T

def sigmoid(x):
    return (1/(1 + np.exp(-x)))
map_pred_pos = f(torch.tensor(positions, device = args.device, dtype = args.dtype)).detach().cpu().numpy().reshape(n_samples, n_samples, 3)

cp = axis[1][0].contourf(X, Y, map_pred_pos[:, :, 0], cmap = plt.get_cmap('Blues'))
divider = make_axes_locatable(axis[1][0])
cax = divider.append_axes('right', size='5%', pad=0.05)
fig.colorbar(cp, cax=cax, orientation='vertical')

cp = axis[1][1].contourf(X, Y, map_pred_pos[:, :, 1], cmap = plt.get_cmap('Oranges'))
divider = make_axes_locatable(axis[1][1])
cax = divider.append_axes('right', size='5%', pad=0.05)
fig.colorbar(cp, cax=cax, orientation='vertical')

cp = axis[1][2].contourf(X, Y, map_pred_pos[:, :, 2], cmap = plt.get_cmap('Greens'))
divider = make_axes_locatable(axis[1][2])
cax = divider.append_axes('right', size='5%', pad=0.05)
fig.colorbar(cp, cax=cax, orientation='vertical')

map_pred = f(torch.tensor(train_dataset.inputs, device = args.device, dtype = args.dtype)).detach().cpu().numpy()
map_pred = np.argmax(map_pred, -1)
axis[0][1].scatter(train_dataset.inputs[:, 0], train_dataset.inputs[:, 1], c = color_map(map_pred.astype(np.int32)), alpha = 0.8, label = "Training Dataset")
axis[0][1].set_title("MAP Predictions")


grid_dataset = Test_Dataset(positions)
grid_loader = torch.utils.data.DataLoader(grid_dataset, batch_size = args.batch_size)


_, lla_var = forward(lla, grid_loader)

#lla_var = lla_var.reshape(n_samples, n_samples, 3, 3)
color_map = plt.get_cmap('coolwarm') 


for i in range(3):
    for j in range(i, 3):

        cp = axis[2+i][j].contourf(X, Y, lla_var[:, i, j].reshape(n_samples, n_samples), cmap = color_map)
        divider = make_axes_locatable(axis[2+i][j])
        cax = divider.append_axes('right', size='5%', pad=0.05)
        fig.colorbar(cp, cax=cax,  orientation='vertical')
        
        axis[2+i][j].scatter(train_dataset.inputs[:, 0], train_dataset.inputs[:, 1],
                        c = plt.get_cmap('tab10')(train_dataset.targets.astype(np.int32)), 
                        alpha = 0.2, label = "Training Dataset")
        
        if i != j:
            cp = axis[2+j][i].contourf(X, Y, lla_var[:, j, i].reshape(n_samples, n_samples),  cmap = color_map)
            divider = make_axes_locatable(axis[j+2][i])
            cax = divider.append_axes('right', size='5%', pad=0.05)
            fig.colorbar(cp, cax=cax, orientation='vertical')
            
            axis[2+j][i].scatter(train_dataset.inputs[:, 0], train_dataset.inputs[:, 1],
                        c = plt.get_cmap('tab10')(train_dataset.targets.astype(np.int32)), 
                        alpha = 0.2, label = "Training Dataset")


plt.savefig("LLA_{}_prior={}.pdf".format(args.dataset_name, str(args.prior_std)), bbox_inches='tight')
plt.show()