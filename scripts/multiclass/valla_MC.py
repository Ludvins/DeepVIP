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
from utils.pytorch_learning import fit_map_crossentropy, fit, forward, score
from scripts.filename import create_file_name
from src.generative_functions import *
from src.valla import VaLLAMultiClassMC
from utils.models import get_mlp, create_ad_hoc_mlp
from utils.dataset import get_dataset
from utils.metrics import SoftmaxClassification
from src.utils import smooth
import matplotlib.pyplot as plt
args = manage_experiment_configuration()

args.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
torch.manual_seed(args.seed)

args.dataset = get_dataset(args.dataset_name)
train_dataset, val_dataset, test_dataset = args.dataset.get_split(
    args.test_size, args.seed + args.split
)

# Initialize DataLoader
train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle = True)
test_loader = DataLoader(test_dataset, batch_size=args.batch_size)


f = get_mlp(
    train_dataset.inputs.shape[1],
    args.dataset.output_dim,
    [200, 200],
    torch.nn.Tanh,
    device=args.device,
    dtype=args.dtype,
)
# Define optimizer and compile model
opt = torch.optim.Adam(f.parameters(), lr=args.MAP_lr, weight_decay=args.weight_decay)
criterion = torch.nn.MSELoss()


try:
    f.load_state_dict(torch.load("weights/multiclass_weights_" + args.dataset_name))
except:
    # Set the number of training samples to generate
    # Train the model
    start = timer()

    loss = fit_map_crossentropy(
        f,
        train_loader,
        opt,
        criterion=torch.nn.CrossEntropyLoss(),
        use_tqdm=True,
        return_loss=True,
        iterations=args.MAP_iterations,
        device=args.device,
    )
    import matplotlib.pyplot as plt
    iters_per_epoch = len(train_loader)
    if len(loss) > iters_per_epoch:
        loss = smooth(np.array(loss), iters_per_epoch)

        plt.plot(loss)
        plt.show()
    print("MAP Loss: ", loss[-1])
    end = timer()
    torch.save(f.state_dict(), "weights/multiclass_weights_" + args.dataset_name)

Z = kmeans2(train_dataset.inputs, args.num_inducing, minit="points", seed=args.seed)[0]

valla = VaLLAMultiClassMC(
    create_ad_hoc_mlp(f),
    Z,
    prior_std=args.prior_std,
    num_data=train_dataset.inputs.shape[0],
    output_dim=train_dataset.output_dim,
    track_inducing_locations=True,
    y_mean=train_dataset.targets_mean,
    y_std=train_dataset.targets_std,
    alpha = args.bb_alpha,
    device=args.device,
    dtype=args.dtype,
    seed = args.seed
    )

# if args.freeze_ll:
#     sparseLA.freeze_ll()


valla.print_variables()

opt = torch.optim.Adam(valla.parameters(recurse=False), lr=args.lr)

start = timer()
loss = fit(
    valla,
    train_loader,
    opt,
    use_tqdm=True,
    return_loss=True,
    iterations=args.iterations,
    device=args.device,
)
end = timer()

iters_per_epoch = len(train_loader)
if len(loss) > iters_per_epoch:

    fig, axis = plt.subplots(3, 1, figsize=(15, 20))
    loss = smooth(np.array(loss), iters_per_epoch)
    ell = smooth(np.array(valla.ell_history), iters_per_epoch)
    kl = smooth(np.array(valla.kl_history), iters_per_epoch)

    axis[0].plot(loss)
    axis[0].set_title("Nelbo")

    axis[1].plot(ell)
    axis[1].set_title("ELL")

    axis[2].plot(kl)
    axis[2].set_title("KL")

    plt.show()


valla.print_variables()

save_str = "VaLLA_MC_dataset={}_M={}".format(
    args.dataset_name, args.num_inducing, args.seed
)


plt.savefig("plots/" + save_str + ".pdf")

test_metrics = score(
    valla,
    test_loader,
    SoftmaxClassification,
    use_tqdm=True,
    device=args.device,
    dtype=args.dtype,
)

test_metrics["prior_std"] = valla.prior_std.detach().numpy()
test_metrics["iterations"] = args.iterations
test_metrics["weight_decay"] = args.weight_decay
test_metrics["dataset"] = args.dataset_name
test_metrics["MAP_iterations"] = args.MAP_iterations
test_metrics["M"] = args.num_inducing

test_metrics["alpha"] = args.bb_alpha
df = pd.DataFrame.from_dict(test_metrics, orient="index").transpose()

print(df)

df.to_csv(
    path_or_buf="results/" + save_str + ".csv",
    encoding="utf-8",
)
