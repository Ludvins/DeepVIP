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
from src.sparseLA import VaLLAMultiClass, GPLLA
from src.likelihood import ARMultiClass, MultiClass, GaussianMultiClass
from src.backpack_interface import BackPackInterface
from utils.models import get_mlp
from utils.dataset import get_dataset, Test_Dataset
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
            args.device, args.dtype)


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


#Z = kmeans2(train_dataset.inputs, args.num_inducing, minit="points", seed=args.seed)

rng = np.random.default_rng(args.seed)
indexes = rng.choice(np.arange(train_dataset.inputs.shape[0]),
                             args.num_inducing, 
                             replace = False)
Z = train_dataset.inputs[indexes]
classes = train_dataset.targets[indexes].flatten()

sparseLA = VaLLAMultiClass(
    f.forward,
    Z, 
    inducing_classes = classes,
    alpha = args.bb_alpha,
    prior_std=args.prior_std,
    likelihood=None,
    num_data = train_dataset.inputs.shape[0],
    output_dim = args.dataset.output_dim,
    backend = BackPackInterface(f, f.output_size),
    track_inducing_locations=True,
    y_mean=train_dataset.targets_mean,
    y_std=train_dataset.targets_std,
    device=args.device,
    dtype=args.dtype,)


sparseLA.print_variables()

opt = torch.optim.Adam(sparseLA.parameters(), lr=args.lr)

path = "weights/valla_multiclass_weights_{}_{}".format(str(args.num_inducing), str(args.iterations))

import os
if os.path.isfile(path):
    print("Pre-trained weights found")

    sparseLA.load_state_dict(torch.load(path))
else:
    print("Pre-trained weights not found")
    start = timer()
    loss = fit(
        sparseLA,
        train_loader,
        opt,
        use_tqdm=True,
        return_loss=True,
        iterations=args.iterations,
        device=args.device,
    )
    end = timer()
    fig, axis = plt.subplots(3, 1, figsize=(15, 20))
    def moving_average(a, n=3):
        border = np.floor(n/2).astype(int)
        new_a = np.zeros(len(a) + 2*border)
        new_a[:border] = a[0]
        new_a[-border:] = a[-1]
        new_a[border:-border] = a
        ret = np.cumsum(new_a, dtype=float)
        ret[n:] = ret[n:] - ret[:-n]
        return ret[n - 1:] / n
    axis[0].plot(loss, color = "blue", alpha = 0.2)
    axis[0].plot(moving_average(loss, 7), color = "blue")
    axis[0].set_title("Nelbo")
    
    axis[1].plot(sparseLA.ell_history, color = "blue", alpha = 0.2)
    axis[1].plot(moving_average(sparseLA.ell_history, 7), color = "blue")
    axis[1].set_title("ELL")

    axis[2].plot(sparseLA.kl_history)
    axis[2].set_title("KL")

    plt.show()
    #plt.clf()

    # torch.save(sparseLA.state_dict(), path)
    

sparseLA.print_variables()




def hessian(x, y):
    #oh = torch.nn.functional.one_hot(y.long().flatten(), args.dataset.classes).type(args.dtype)
    out = torch.nn.Softmax(dim = -1)(f(x))
    a = torch.einsum("na, nb -> abn", out, out)
    b = torch.diag_embed(out).permute(1,2,0)
    #b = torch.sum(out * oh, -1)
    return - a + b

lla = GPLLA(f, 
            prior_std=args.prior_std,
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
fig, axis = plt.subplots(4, 3, figsize=(15, 20))

color_map = plt.get_cmap('tab10') 
axis[0][0].scatter(train_dataset.inputs[:, 0], train_dataset.inputs[:, 1], c = color_map(train_dataset.targets.astype(np.int32)), alpha = 0.8, label = "Training Dataset")
axis[0][0].set_title("Training Dataset")
xlims = axis[0][0].get_xlim()
ylims = axis[0][0].get_ylim()



n_samples = 50
x_vals = np.linspace(xlims[0], xlims[1], n_samples)
y_vals = np.linspace(ylims[0], ylims[1], n_samples)
X, Y = np.meshgrid(x_vals, y_vals)
positions = np.vstack([X.ravel(), Y.ravel()]).T

def sigmoid(x):
    return (1/(1 + np.exp(-x)))
map_pred_pos = f(torch.tensor(positions, device = args.device, dtype = args.dtype)).detach().cpu().numpy().reshape(n_samples, n_samples, 3)

axis[0][1].contourf(X, Y, sigmoid(map_pred_pos[:, :, 0]), cmap = plt.get_cmap('Blues'), alpha = 0.33)
axis[0][1].contourf(X, Y, sigmoid(map_pred_pos[:, :, 1]), cmap = plt.get_cmap('Oranges'), alpha = 0.33)
axis[0][1].contourf(X, Y, sigmoid(map_pred_pos[:, :, 2]), cmap = plt.get_cmap('Greens'), alpha = 0.33)

map_pred = f(torch.tensor(train_dataset.inputs, device = args.device, dtype = args.dtype)).detach().cpu().numpy()
map_pred = np.argmax(map_pred, -1)
axis[0][1].scatter(train_dataset.inputs[:, 0], train_dataset.inputs[:, 1], c = color_map(map_pred.astype(np.int32)), alpha = 0.8, label = "Training Dataset")
axis[0][1].set_title("MAP Predictions")

inducing_history = np.stack(sparseLA.inducing_history)

colors = cm.rainbow(np.linspace(0, 1, inducing_history.shape[1]))
sizes = np.linspace(1, 10, inducing_history.shape[0])
for i in np.arange(start = 0, stop = inducing_history.shape[0], step = np.max([inducing_history.shape[0]//100, 1])):
    axis[0][2].scatter(inducing_history[i, :, 0], inducing_history[i, :, 1], c = colors, s= sizes[i],  alpha = 0.8)
axis[0][2].set_ylim(axis[0][1].get_ylim()[0], axis[0][1].get_ylim()[1])
axis[0][2].set_xlim(axis[0][1].get_xlim()[0], axis[0][1].get_xlim()[1])

grid_dataset = Test_Dataset(positions)
grid_loader = torch.utils.data.DataLoader(grid_dataset, batch_size = args.batch_size)


_, valla_var = forward(sparseLA, grid_loader)
_, lla_var = forward(lla, grid_loader)
#lla_var = lla_var.reshape(n_samples, n_samples, 3, 3)
color_map = plt.get_cmap('coolwarm') 


for i in range(3):
    for j in range(i, 3):

        cp = axis[1+i][j].contourf(X, Y, valla_var[:, i, j].reshape(n_samples, n_samples), cmap = color_map)
        divider = make_axes_locatable(axis[i+1][j])
        cax = divider.append_axes('right', size='5%', pad=0.05)
        fig.colorbar(cp, cax=cax,  orientation='vertical')
        if i != j:
            cp = axis[1+j][i].contourf(X, Y, valla_var[:, j, i].reshape(n_samples, n_samples),  cmap = color_map)
            divider = make_axes_locatable(axis[j+1][i])
            cax = divider.append_axes('right', size='5%', pad=0.05)
            fig.colorbar(cp, cax=cax, orientation='vertical')



plt.savefig("VaLLA_AR_{}_M={}_prior={}.pdf".format(args.dataset_name, args.num_inducing, args.prior_std), bbox_inches='tight')

def kl(sigma1, sigma2):
    L1 = np.linalg.cholesky(sigma1 + 1e-3 * np.eye(sigma1.shape[-1]))
    L2 = np.linalg.cholesky(sigma2 + 1e-3 * np.eye(sigma2.shape[-1]))
    M = np.linalg.solve(L2, L1)
    a = 0.5 * np.sum(M**2)
    b = - 0.5 * sigma1.shape[-1] 
    c = np.sum(np.log(np.diagonal(L2, axis1= 1, axis2 = 2))) - np.sum(np.log(np.diagonal(L1, axis1= 1, axis2 = 2)))
    return a + b + c
    
def w2(sigma1, sigma2):
    a = sigma1 @ sigma2
    u, s, vh = np.linalg.svd(a, full_matrices=True)
    sqrt = u * np.sqrt(s)[..., np.newaxis] @ vh
    w = np.trace(sigma1, axis1 = 1, axis2 = 2) + np.trace(sigma2, axis1 = 1, axis2 = 2) - 2*np.trace(sqrt, axis1 = 1, axis2 = 2)
    return np.sum(w)

KL1 = kl(valla_var, lla_var)/(n_samples*n_samples)
KL2 = kl(lla_var, valla_var)/(n_samples*n_samples)
W2 = w2(valla_var, lla_var)


MAE = np.mean(np.abs(valla_var - lla_var))

d = {
    "M": args.num_inducing,
    "seed": args.seed,
    "KL": 0.5*KL1 + 0.5*KL2,
    "W2": W2,
    "MAE": MAE,
    "map_iterations": args.MAP_iterations,
    "prior_std": args.prior_std,
    "iterations": args.iterations
}

df = pd.DataFrame.from_dict(d, orient="index").transpose()

print(df)

df.to_csv(
    path_or_buf="results/VaLLA_MCGaussian_dataset={}_M={}_prior={}_seed={}.csv".format(args.dataset_name,args.num_inducing, str(args.prior_std), args.seed),
    encoding="utf-8",
)