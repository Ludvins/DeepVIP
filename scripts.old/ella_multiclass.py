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
from utils.pytorch_learning import (
    fit_map_crossentropy,
    fit,
    predict,
    forward,
    score,
    acc_multiclass,
)
from scripts.filename import create_file_name
from src.generative_functions import *
from src.sparseLA import ELLA_Optimized
from src.likelihood import GaussianMultiClassSubset, MultiClass
from src.backpack_interface import BackPackInterface
from utils.metrics import SoftmaxClassification

from utils.models import get_mlp, create_ad_hoc_mlp
from utils.dataset import get_dataset

args = manage_experiment_configuration()

args.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
torch.manual_seed(args.seed)


train_dataset, train_test_dataset, test_dataset = args.dataset.get_split(
    args.test_size, args.seed + args.split
)

# Initialize DataLoader
train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
train_test_loader = DataLoader(train_test_dataset, batch_size=args.batch_size)
test_loader = DataLoader(test_dataset, batch_size=args.batch_size)


f = get_mlp(
    train_dataset.inputs.shape[1],
    args.dataset.output_dim,
    [1024, 512, 512, 256],
    torch.nn.Tanh,
    device=args.device,
    dtype=args.dtype,
)

if args.weight_decay != 0 and args.prior_std == 0:
    args.prior_std = np.sqrt(1 / (len(train_dataset) * args.weight_decay))

if args.weight_decay == 0:
    args.weight_decay = 1 / (len(train_dataset) * args.prior_std**2)



# Define optimizer and compile model
opt = torch.optim.Adam(f.parameters(), lr=args.MAP_lr, weight_decay=args.weight_decay)
criterion = torch.nn.CrossEntropyLoss()
str_name = (
    "weights/multiclass_weights_" + args.dataset_name + "_" + str(args.weight_decay)
)

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
        criterion=criterion,
        use_tqdm=True,
        return_loss=True,
        iterations=args.MAP_iterations,
        device=args.device,
        dtype=args.dtype,
    )
    plt.plot(loss)

    plt.show()
    print("MAP Loss: ", loss[-1])
    end = timer()

    train_acc = acc_multiclass(
        f, train_loader, use_tqdm=True, device=args.device, dtype=args.dtype
    )

    test_acc = acc_multiclass(
        f, test_loader, use_tqdm=True, device=args.device, dtype=args.dtype
    )

    print("Train acc: ", train_acc)
    print("Test acc: ", test_acc)

    torch.save(f.state_dict(), str_name)


def hessian(x, y):
    # oh = torch.nn.functional.one_hot(y.long().flatten(), args.dataset.classes).type(args.dtype)
    out = torch.nn.Softmax(dim=-1)(f(x))
    a = torch.einsum("na, nb -> abn", out, out)
    b = torch.diag_embed(out).permute(1, 2, 0)
    # b = torch.sum(out * oh, -1)
    return -a + b


ella = ELLA_Optimized(
    create_ad_hoc_mlp(f),
    f.output_size,
    args.num_inducing,
    np.min([args.num_inducing, 20]),
    prior_std=args.prior_std,
    likelihood_hessian=lambda x, y: hessian(x, y),
    likelihood=MultiClass(
        num_classes=args.dataset.classes, device=args.device, dtype=args.dtype
    ),
    seed=args.seed,
    device=args.device,
    dtype=args.dtype,
)


ella.fit_loader(
    torch.tensor(train_dataset.inputs, device=args.device, dtype=args.dtype),
    torch.tensor(train_dataset.targets, device=args.device, dtype=args.dtype),
    train_loader,
)


print("Training done")


test_metrics = score(
    ella,
    test_loader,
    SoftmaxClassification,
    use_tqdm=True,
    device=args.device,
    dtype=args.dtype,
)
test_metrics["prior_std"] = args.prior_std
test_metrics["weight_decay"] = args.weight_decay
test_metrics["dataset"] = args.dataset_name
test_metrics["MAP_iterations"] = args.MAP_iterations
test_metrics["M"] = ella.M
test_metrics["K"] = ella.K

df = pd.DataFrame.from_dict(test_metrics, orient="index").transpose()
print(df)


df.to_csv(
    "results/ELLA_{}_MAP_iterations={}_prior_std={}_K={}_M={}.csv".format(
        args.dataset_name,
        str(args.MAP_iterations),
        str(args.prior_std),
        str(ella.K),
        str(ella.M),
    ),
    index=False,
)
