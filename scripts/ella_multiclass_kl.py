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


def s():
    return torch.nn.Softmax(dim = -1)

print(args.dataset.output_dim)

f = get_mlp(train_dataset.inputs.shape[1], args.dataset.output_dim, 
            [50, 50], torch.nn.Tanh, s,
            args.device, args.dtype)

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
    out = f(x)
    a = torch.einsum("na, nb -> abn", out, out)
    b = torch.diag_embed(out).permute(1,2,0)
    #b = torch.sum(out * oh, -1)
    return - a + b


ella = ELLA(f[:-1], 
            args.num_inducing,
            np.min([args.num_inducing, 20]),
            prior_std = 2.7209542,
            likelihood_hessian=lambda x,y: hessian(x, y),
            likelihood=MultiClass(num_classes = args.dataset.classes,
                          device=args.device, 
                        dtype = args.dtype), 
            backend = BackPackInterface(f, "classification"),
            seed = args.seed,
            device = args.device,
            dtype = args.dtype)


ella.fit(torch.tensor(train_dataset.inputs, device = args.device, dtype = args.dtype),
         torch.tensor(train_dataset.targets, device = args.device, dtype = args.dtype))


lla = GPLLA(f[:-1], 
            prior_std = 2.7209542,
            likelihood_hessian=lambda x,y: hessian(x, y),
            likelihood=MultiClass(num_classes = args.dataset.classes,
                          device=args.device, 
                        dtype = args.dtype), 
            backend = BackPackInterface(f, "classification"),
            device = args.device,
            dtype = args.dtype)


lla.fit(torch.tensor(train_dataset.inputs, device = args.device, dtype = args.dtype),
        torch.tensor(train_dataset.targets, device = args.device, dtype = args.dtype))


n_samples = 50
x_vals = np.linspace(-3, 3, n_samples)
y_vals = np.linspace(-3, 3, n_samples)
X, Y = np.meshgrid(x_vals, y_vals)
positions = np.vstack([X.ravel(), Y.ravel()]).T

grid_dataset = Test_Dataset(positions)
grid_loader = torch.utils.data.DataLoader(grid_dataset, batch_size = args.batch_size)


_, ella_var = forward(ella, grid_loader)
_, lla_var = forward(lla, grid_loader)

MAE = np.mean(np.abs(ella_var - lla_var))

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

kl_valla_lla = kl(ella_var, lla_var)
kl_lla_valla = kl(lla_var, ella_var)

print("MAE: ", MAE)
print("KL(VaLLA|LLA): ", kl_valla_lla)
print("KL(LLA|VaLLA): ", kl_lla_valla)
print("KL: ", 0.5* kl_lla_valla + 0.5*kl_valla_lla)

print("W2: ", w2(ella_var, lla_var))
