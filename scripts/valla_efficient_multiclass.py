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
from utils.pytorch_learning import fit_map_crossentropy, fit, predict, forward, score, acc_multiclass
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
            [100, 100], torch.nn.ReLU,
            device = args.device, dtype = args.dtype)

# Define optimizer and compile model
opt = torch.optim.Adam(f.parameters(), lr=args.MAP_lr)
criterion = torch.nn.CrossEntropyLoss()

try:
    f.load_state_dict(torch.load("weights/multiclass_weights_"+args.dataset_name))
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
    plt.plot(loss)
    
    plt.show()
    print("MAP Loss: ", loss[-1])
    end = timer()
    
    train_acc = acc_multiclass(f, train_loader, use_tqdm=True, 
        device=args.device,
        dtype = args.dtype)
    
    test_acc = acc_multiclass(f, test_loader, use_tqdm=True, 
        device=args.device,
        dtype = args.dtype)
    
    print("Train acc: ", train_acc)
    print("Test acc: ", test_acc)

    torch.save(f.state_dict(), "weights/multiclass_weights_"+args.dataset_name)


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





_, valla_var = forward(sparseLA, test_loader)
_, lla_var = forward(lla, test_loader)

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

KL1 = kl(valla_var, lla_var)/(test_dataset.inputs.shape[0])
KL2 = kl(lla_var, valla_var)/(test_dataset.inputs.shape[0])
W2 = w2(valla_var, lla_var)


MAE = np.mean(np.abs(valla_var - lla_var))

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
    path_or_buf="results/VaLLA_dataset={}_M={}_prior={}_seed={}.csv".format(args.dataset_name,args.num_inducing, str(args.prior_std), args.seed),
    encoding="utf-8",
)