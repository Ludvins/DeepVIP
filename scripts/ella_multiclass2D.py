from datetime import datetime
import numpy as np
import torch
import pandas as pd
from torch.utils.data import DataLoader
import sys
from time import process_time as timer
from properscoring import crps_gaussian
from scipy.special import expit


from scipy.cluster.vq import kmeans2
sys.path.append(".")
from mpl_toolkits.axes_grid1 import make_axes_locatable

from utils.process_flags import manage_experiment_configuration
from utils.pytorch_learning import fit_map_crossentropy, fit, predict, forward, score
from scripts.filename import create_file_name
from src.generative_functions import *
from src.sparseLA import VaLLASampling, GPLLA, ELLA
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



print(args.dataset.output_dim)

f = get_mlp(train_dataset.inputs.shape[1], args.dataset.output_dim, 
            [50, 50], torch.nn.Tanh,
            device = args.device, dtype = args.dtype)

print(f)

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


ella = ELLA(f, 
            f.output_size,
            args.num_inducing,
            np.min([args.num_inducing, 20]),
            prior_std = args.prior_std,
            likelihood_hessian=lambda x,y: hessian(x, y),
            likelihood=MultiClass(num_classes = args.dataset.classes,
                          device=args.device, 
                        dtype = args.dtype), 
            backend = BackPackInterface(f, f.output_size),
            seed = args.seed,
            device = args.device,
            dtype = args.dtype)


ella.fit(torch.tensor(train_dataset.inputs, device = args.device, dtype = args.dtype),
         torch.tensor(train_dataset.targets, device = args.device, dtype = args.dtype))


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


import matplotlib.pyplot as plt
import matplotlib.cm as cm



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


axis[0][2].scatter(ella.Xz[:, 0], ella.Xz[:, 1], alpha = 0.8)
axis[0][2].set_ylim(axis[0][1].get_ylim()[0], axis[0][1].get_ylim()[1])
axis[0][2].set_xlim(axis[0][1].get_xlim()[0], axis[0][1].get_xlim()[1])

grid_dataset = Test_Dataset(positions)
grid_loader = torch.utils.data.DataLoader(grid_dataset, batch_size = args.batch_size)


_, ella_var = forward(ella, grid_loader)
_, lla_var = forward(lla, grid_loader)

#lla_var = lla_var.reshape(n_samples, n_samples, 3, 3)
color_map = plt.get_cmap('coolwarm') 


for i in range(3):
    for j in range(i, 3):

        cp = axis[1+i][j].contourf(X, Y, ella_var[:, i, j].reshape(n_samples, n_samples), cmap = color_map)
        divider = make_axes_locatable(axis[i+1][j])
        cax = divider.append_axes('right', size='5%', pad=0.05)
        fig.colorbar(cp, cax=cax,  orientation='vertical')
        if i != j:
            cp = axis[1+j][i].contourf(X, Y, ella_var[:, j, i].reshape(n_samples, n_samples),  cmap = color_map)
            divider = make_axes_locatable(axis[j+1][i])
            cax = divider.append_axes('right', size='5%', pad=0.05)
            fig.colorbar(cp, cax=cax, orientation='vertical')


plt.savefig("ELLA_{}_M={}_prior={}.pdf".format(args.dataset_name, args.num_inducing, args.prior_std), bbox_inches='tight')


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

KL1 = kl(ella_var, lla_var)/(n_samples*n_samples)
KL2 = kl(lla_var, ella_var)/(n_samples*n_samples)
W2 = w2(ella_var, lla_var)


MAE = np.mean(np.abs(ella_var - lla_var))

d = {
    "M": args.num_inducing,
    "seed": args.seed,
    "KL": 0.5*KL1 + 0.5*KL2,
    "W2": W2,
    "MAE": MAE,
    "map_iterations": args.MAP_iterations,
    "prior_std": args.prior_std
}

df = pd.DataFrame.from_dict(d, orient="index").transpose()

print(df)

df.to_csv(
    path_or_buf="results/ELLA_dataset={}_M={}_prior={}_seed={}.csv".format(args.dataset_name,args.num_inducing, str(args.prior_std), args.seed),
    encoding="utf-8",
)