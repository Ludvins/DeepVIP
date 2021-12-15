import torch
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader
import sys

sys.path.append(".")

from src.dvip import DVIP_Base
from src.layers_init import init_layers
from src.likelihood import Gaussian
from utils.dataset import Test_Dataset, Training_Dataset
from utils.metrics import Metrics
from utils.process_flags import manage_experiment_configuration
from utils.pytorch_learning import fit, score

args = manage_experiment_configuration()

torch.manual_seed(2147483647)

# CUDA for PyTorch
use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")
torch.backends.cudnn.benchmark = True
vars(args)["device"] = device

n_splits = 10
test_history = Metrics().get_dict()
kfold = KFold(n_splits, shuffle=True, random_state=2147483647)

for train_indexes, test_indexes in kfold.split(args.dataset.inputs):
    train_dataset = Training_Dataset(
        args.dataset.inputs[train_indexes],
        args.dataset.targets[train_indexes],
        verbose=False,
    )
    test_dataset = Test_Dataset(args.dataset.inputs[test_indexes],
                                args.dataset.targets[test_indexes],
                                train_dataset.inputs_mean,
                                train_dataset.inputs_std)

    # Get VIP layers
    layers = init_layers(train_dataset.inputs, train_dataset.output_dim,
                         **vars(args))

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size)
    val_loader = DataLoader(test_dataset, batch_size=args.batch_size)

    # Instantiate Likelihood
    ll = Gaussian()

    # Create DVIP object
    dvip = DVIP_Base(
        ll,
        layers,
        len(train_dataset),
        num_samples=args.num_samples_train,
        y_mean=train_dataset.targets_mean,
        y_std=train_dataset.targets_std,
        dtype=args.dtype,
        device=args.device,
    )
    # dvip.freeze_prior()

    # Define optimizer and compile model
    opt = torch.optim.Adam(dvip.parameters(), lr=args.lr)

    # Perform training
    dvip.num_samples = args.num_samples_train
    train_hist = fit(
        dvip,
        train_loader,
        opt,
        # val_generator=val_loader,
        epochs=args.epochs,
        verbose=0,
        device=args.device)
    dvip.num_samples = args.num_samples_test
    test_metrics = score(dvip, val_loader, device=args.device)

    test_history = {
        key: test_history[key] + test_metrics[key]
        for key in test_history.keys()
    }
    print("FOLD RESULTS: ")
    print("\t - NELBO: {}".format(test_metrics["LOSS"]))
    print("\t - NLL: {}".format(test_metrics["NLL"]))
    print("\t - RMSE: {}".format(test_metrics["RMSE"]))

print("TEST RESULTS: ")
print("\t - NELBO: {}".format(test_history["LOSS"] / n_splits))
print("\t - NLL: {}".format(test_history["NLL"] / n_splits))
print("\t - RMSE: {}".format(test_history["RMSE"] / n_splits))
""" 
    # Predict Train and Test
    train_mean, train_var = predict(dvip, predict_loader)
    train_prediction_mean, train_prediction_var = dvip.get_predictive_results(
        train_mean, train_var)

    # Change MC samples for test
    dvip.num_samples = args.num_samples_test
    test_mean, test_var = predict(dvip, test_loader)
    test_prediction_mean, test_prediction_var = dvip.get_predictive_results(
        test_mean, test_var)

    print(args.X_train.shape)
    print(train_mean.shape)
    import matplotlib.pyplot as plt

    # plot it
    fig = plt.figure(figsize=(20, 10))
    axs = []
    axs.append(fig.add_subplot(3, 5, 1))
    axs.append(fig.add_subplot(3, 5, 2))
    axs.append(fig.add_subplot(3, 5, 3))
    axs.append(fig.add_subplot(3, 5, 4))

    axs.append(fig.add_subplot(3, 5, 6))
    axs.append(fig.add_subplot(3, 5, 7))
    axs.append(fig.add_subplot(3, 5, 8))
    axs.append(fig.add_subplot(3, 5, 9))

    axs.append(fig.add_subplot(3, 5, 11))
    axs.append(fig.add_subplot(3, 5, 12))
    axs.append(fig.add_subplot(3, 5, 13))
    axs.append(fig.add_subplot(3, 5, 14))

    axs.append(fig.add_subplot(1, 5, 5))

    for i in range(args.X_train.shape[1]):
        axs[i].scatter(args.X_train[:, i],
                       args.y_train,
                       color="teal",
                       label="Original data",
                       s=1.5)
        axs[i].scatter(args.X_train[:, i],
                       train_mean.flatten(),
                       color="darkorange",
                       label="Prediction",
                       s=1.5)
        axs[i].legend()
    plt.savefig("2_layers1-1.svg", format="svg")
    plt.savefig("2_layers1-1.png", format="png")
    plt.show()
 """
