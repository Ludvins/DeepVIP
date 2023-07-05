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
from src.sparseLA import VaLLAMultiClassSubsetOptimized, GPLLA
from src.likelihood import GaussianMultiClassSubset, MultiClass
from src.backpack_interface import BackPackInterface
from utils.metrics import SoftmaxClassification

from utils.models import get_mlp, create_ad_hoc_mlp
from utils.dataset import get_dataset
from src.utils import smooth

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


f = get_mlp(train_dataset.inputs.shape[1], args.dataset.output_dim, [200, 200], torch.nn.Tanh,
            device = args.device, dtype = args.dtype)

if args.weight_decay != 0:
    args.prior_std = np.sqrt(1/(len(train_dataset) * args.weight_decay))
    
if args.weight_decay == 0:
    args.weight_decay =  1/(len(train_dataset) * args.prior_std**2)
    
    
# Define optimizer and compile model
opt = torch.optim.Adam(f.parameters(), lr=args.MAP_lr, weight_decay=args.weight_decay)
criterion = torch.nn.CrossEntropyLoss()
str_name = "weights/multiclass_weights_"+args.dataset_name+"_"+str(args.weight_decay)

print(str_name)
try:
    f.load_state_dict(torch.load(str_name))
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

    torch.save(f.state_dict(), str_name)

#Z = kmeans2(train_dataset.inputs, args.num_inducing, minit="points", seed=args.seed)

rng = np.random.default_rng(args.seed)
indexes = rng.choice(np.arange(train_dataset.inputs.shape[0]),
                             args.num_inducing, 
                             replace = False)
Z = train_dataset.inputs[indexes]
classes = train_dataset.targets[indexes].flatten()

sparseLA = VaLLAMultiClassSubsetOptimized(
    create_ad_hoc_mlp(f),
    Z, 
    n_classes_subsampled = args.sub_classes,
    inducing_classes = classes,
    prior_std=args.prior_std,
    likelihood=GaussianMultiClassSubset(device=args.device, 
                        dtype = args.dtype), 
    num_data = train_dataset.inputs.shape[0],
    output_dim = args.dataset.output_dim,
    backend = BackPackInterface(f, f.output_size),
    fix_inducing_locations=False,
    track_inducing_locations=False,
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
    iters_per_epoch = len(train_loader)
    loss = smooth(np.array(loss),  iters_per_epoch,  window='flat')
    ell = smooth(np.array(sparseLA.ell_history), iters_per_epoch, window = "flat")
    kl = smooth(np.array(sparseLA.kl_history), iters_per_epoch, window = "flat")

    
    axis[0].plot(loss[::iters_per_epoch])
    axis[0].set_title("Nelbo")
    
    axis[1].plot(ell[::iters_per_epoch])    
    axis[1].set_title("ELL")

    axis[2].plot(kl[::iters_per_epoch])
    axis[2].set_title("KL")

    plt.show()
    #plt.clf()

    #torch.save(sparseLA.state_dict(), path)
    

sparseLA.print_variables()



test_metrics = score(sparseLA, test_loader, SoftmaxClassification, use_tqdm=True, device = args.device, dtype = args.dtype)
test_metrics["prior_std"] = args.prior_std
test_metrics["iterations"] = args.iterations
test_metrics["weight_decay"] = args.weight_decay
test_metrics["dataset"] = args.dataset_name
test_metrics["MAP_iterations"] = args.MAP_iterations
test_metrics["M"] = args.num_inducing
test_metrics["Classes"] = args.sub_classes

df = pd.DataFrame.from_dict(test_metrics, orient="index").transpose()
print(df)


df.to_csv("results/VaLLA_{}_{}_MAP_it={}_it={}_prior={}_M={}.csv".format(
    args.dataset_name, str(args.sub_classes), str(args.MAP_iterations), 
    str(args.iterations), str(args.prior_std), 
    str(args.num_inducing)), 
          index = False)
