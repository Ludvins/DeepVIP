import torch
import pandas as pd
from torch.utils.data import DataLoader
import sys
from time import process_time as timer

from scipy.cluster.vq import kmeans2

sys.path.append(".")

from utils.process_flags import manage_experiment_configuration
from utils.pytorch_learning import fit_map, fit, score
from src.valla import VaLLARegressionRBF
from src.backpack_interface import BackPackInterface
from utils.models import get_mlp, create_ad_hoc_mlp
from utils.dataset import get_dataset
from utils.metrics import Regression

args = manage_experiment_configuration()

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
    args.dataset.input_dim,
    args.dataset.output_dim,
    args.net_structure,
    args.activation,
    device=args.device,
    dtype=args.dtype,
)



# Define optimizer and compile model
opt = torch.optim.Adam(f.parameters(), lr=args.MAP_lr, weight_decay=args.weight_decay)
criterion = torch.nn.MSELoss()


try:
    f.load_state_dict(torch.load("weights/regression_weights_" + args.dataset_name))
except:
    # Set the number of training samples to generate
    # Train the model
    start = timer()

    loss = fit_map(
        f,
        train_loader,
        opt,
        criterion=torch.nn.MSELoss(),
        use_tqdm=True,
        return_loss=True,
        iterations=args.MAP_iterations,
        device=args.device,
    )
    print("MAP Loss: ", loss[-1])
    end = timer()
    torch.save(f.state_dict(), "weights/regression_weights_" + args.dataset_name)


Z = kmeans2(train_dataset.inputs, args.num_inducing, minit="points", seed=args.seed)[0]

valla = VaLLARegressionRBF(
    create_ad_hoc_mlp(f),
    Z,
    prior_std=args.prior_std,
    num_data=train_dataset.inputs.shape[0],
    output_dim=1,
    log_variance=args.ll_log_var,
    track_inducing_locations=False,
    trainable_prior = not args.fixed_prior,
    y_mean=train_dataset.targets_mean,
    y_std=train_dataset.targets_std,
    alpha=args.bb_alpha,
    device=args.device,
    dtype=args.dtype,
)


valla.print_variables()

opt = torch.optim.Adam(valla.parameters(recurse=False), lr=args.lr)

start = timer()
loss = fit(
    valla,
    train_loader,
    opt,
    use_tqdm=args.verbose,
    return_loss=True,
    iterations=args.iterations,
    device=args.device,
)
end = timer()

valla.print_variables()

fp = "_fixed_prior" if args.fixed_prior else ""

save_str = "VaLLA_RBF_dataset={}_M={}_seed={}_alpha={}{}".format(
    args.dataset_name, args.num_inducing, args.seed, str(args.bb_alpha),fp
)

test_metrics = score(
    valla,
    test_loader,
    Regression,
    use_tqdm=args.verbose,
    device=args.device,
    dtype=args.dtype,
    ll_var=valla.log_variance.exp().detach(),
)
test_metrics["prior_std"] = valla.prior_std.detach().numpy()
test_metrics["log_variance"] = valla.log_variance.detach().numpy()
test_metrics["iterations"] = args.iterations
test_metrics["weight_decay"] = args.weight_decay
test_metrics["dataset"] = args.dataset_name
test_metrics["MAP_iterations"] = args.MAP_iterations
test_metrics["M"] = args.num_inducing
test_metrics["seed"] = args.seed
test_metrics["time"] = end-start

test_metrics["alpha"] = args.bb_alpha
df = pd.DataFrame.from_dict(test_metrics, orient="index").transpose()


print(df)

df.to_csv(
    path_or_buf="results/" + save_str + ".csv",
    encoding="utf-8",
)
