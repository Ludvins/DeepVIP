import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf


def check_data(X, y, verbose=0):
    if X.shape[0] != y.shape[0]:
        print("Labels and features differ in the number of samples")
        return

    # Compute data information
    n_samples = X.shape[0]
    input_dim = X.shape[1]
    output_dim = y.shape[1]
    y_mean = np.mean(y)
    y_std = np.std(y)

    if verbose > 0:
        print("Number of samples: ", n_samples)
        print("Input dimension: ", input_dim)
        print("Label dimension: ", output_dim)
        print("Labels mean value: ", y_mean)
        print("Labels standard deviation: ", y_std)

    return n_samples, input_dim, output_dim, y_mean, y_std


def build_plot_name(
    vip_layers,
    bnn_structure,
    input_dim,
    output_dim,
    epochs,
    n_samples,
    dataset,
    name_flag,
):
    # Create title name
    dims = np.concatenate(
        ([input_dim], np.ones(vip_layers, dtype=int) * [output_dim])
    )

    dims_name = "-".join(map(str, dims))
    bnn_name = "-".join(map(str, bnn_structure))
    path = "plots/{}_{}_layers={}_bnn={}_epochs={}_batchsize={}.svg".format(
        dataset, name_flag, dims_name, bnn_name, epochs, n_samples
    )
    title = "Layers: {}({})  BNN: {}".format(vip_layers, dims_name, bnn_name)
    return title, path


def plot_train_test(
    train_pred,
    test_pred,
    X_train,
    y_train,
    X_test,
    y_test=None,
    train_prior_samples=None,
    test_prior_samples=None,
    title=None,
    path=None,
):
    mean_train, std_train = train_pred
    mean_test, std_test = test_pred

    fig, ax = plt.subplots(
        2, 2, gridspec_kw={"height_ratios": [3, 1]}, figsize=(20, 10)
    )

    plt.suptitle(title)

    plot_results(
        X=X_train.flatten(),
        mean=mean_train.flatten(),
        std=std_train.flatten(),
        y=y_train.flatten(),
        prior_samples=train_prior_samples,
        ax=ax.T[0],
    )

    if y_test is not None:
        y_test.flatten()

    plot_results(
        X=X_test.flatten(),
        mean=mean_test.flatten(),
        std=std_test.flatten(),
        y=y_test,
        prior_samples=test_prior_samples,
        ax=ax.T[1],
    )

    ax[0][0].set_title("Training results")
    ax[0][1].set_title("Test results")
    plt.savefig(path, format="svg")
    plt.show()


def scatter_data(X, y, label=None, color=None, s=1.0, alpha=1.0, ax=None):
    if ax is None:
        fig, ax = plt.subplots()

    ax.scatter(X, y, color=color, label=label, s=s, alpha=alpha)
    return ax


def plot_prediction(
    X,
    mean,
    std=None,
    label=None,
    mean_color=None,
    std_color=None,
    alpha=1.0,
    ax=None,
):

    if ax is None:
        fig, ax = plt.subplots()

    sort = np.argsort(X)
    X = X[sort]
    mean = mean[sort]

    ax.plot(X, mean, color=mean_color, alpha=alpha, label=label)
    if std is not None:
        std = std[sort]
        ax.fill_between(
            X, mean - 2 * std, mean + 2 * std, color=std_color, alpha=alpha / 2
        )

    return ax


def plot_standard_deviation(
    X, std, color=None, alpha=1.0, label=None, ax=None
):
    if ax is None:
        fig, ax = plt.subplots()

    sort = np.argsort(X)
    X = X[sort]
    std = std[sort]
    ax.fill_between(
        X, np.zeros_like(std), std, color=color, label=label, alpha=alpha
    )

    return ax


def plot_results(X, mean, std, y=None, prior_samples=None, ax=None):
    if ax is None:
        _, ax = plt.subplots(2, 1, gridspec_kw={"height_ratios": [3, 1]})

    if y is not None:
        scatter_data(X, y, label="Points", color="blue", ax=ax[0])

    plot_prediction(
        X,
        mean,
        std,
        label="VIP Predictive Mean",
        mean_color="#029386",
        std_color="#cfe6fc",
        ax=ax[0],
    )

    plot_standard_deviation(
        X,
        std,
        color="#cfe6fc",
        alpha=0.8,
        label="VIP Prediction Standard Deviation",
        ax=ax[1],
    )

    if prior_samples is not None:
        for i in range(prior_samples.shape[0]):
            plot_prediction(
                X,
                prior_samples[i].numpy().flatten(),
                label="Prior samples" if i == 0 else "",
                mean_color="red",
                alpha=0.1,
                ax=ax[0],
            )

    ax[0].legend()
    ax[1].legend()
