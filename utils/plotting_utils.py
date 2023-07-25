import matplotlib.pyplot as plt
import numpy as np

from scripts.filename import create_file_name


def build_plot_name(
    vip_layers,
    bnn_structure,
    activation_str,
    epochs,
    batch_size,
    dataset_name,
    name_flag,
    genf,
    fix_prior_noise,
    freeze_prior,
    freeze_posterior,
    **kwargs
):
    """Generates the title and the path of the figure using the configuration
    of the experiment.
    """

    dims_name = "-".join(map(str, vip_layers))
    model_name = genf
    if genf == "BNN":
        model_name += " " + "-".join(map(str, bnn_structure)) + " " + activation_str

    path = "plots/{}_{}_layers={}_bnn={}_epochs={}_batchsize={}".format(
        dataset_name, name_flag, dims_name, model_name, epochs, batch_size
    )
    if fix_prior_noise:
        path = path + "_fixed_noise"
    if freeze_posterior:
        path = path + "_no_posterior"
    if freeze_prior:
        path = path + "_no_prior"
    title = "Layers: {}({}) {}".format(vip_layers, dims_name, model_name)
    return title, path


def plot_train_test(
    train_mixture_means,
    train_prediction_mean,
    train_prediction_sqrt,
    test_mixture_means,
    test_prediction_mean,
    test_prediction_sqrt,
    X_train,
    y_train,
    X_test,
    y_test=None,
    train_prior_samples=None,
    test_prior_samples=None,
    title=None,
    path=None,
    show=True,
):
    """
    Generates a plot consisting in two subplots, one with the training
    results and one with the test results.
    """

    _, ax = plt.subplots(2, 2, gridspec_kw={"height_ratios": [3, 1]}, figsize=(20, 10))

    plt.suptitle(title)

    # Plot the training results.
    ax[0][0].set_title("Training results")
    plot_results(
        X=X_train,
        means=train_mixture_means,
        predictive_mean=train_prediction_mean,
        predictive_std=train_prediction_sqrt,
        y=y_train,
        prior_samples=train_prior_samples,
        ax=ax.T[0],
    )

    # Plot the test results
    ax[0][1].set_title("Test results")
    plot_results(
        X=X_test,
        means=test_mixture_means,
        predictive_mean=test_prediction_mean,
        predictive_std=test_prediction_sqrt,
        y=y_test,
        prior_samples=test_prior_samples,
        ax=ax.T[1],
    )

    # Save and show the figure
    plt.savefig(path + ".svg", format="svg")
    plt.savefig(path + ".png", format="png")
    if show:
        plt.show()
    plt.close()


def plot_results(
    X,
    means,
    predictive_mean,
    predictive_std,
    y=None,
    prior_samples=None,
    ax=None,
):
    """Makes a plot consisting in two subplots joined vertically.
    The upper one shows the points and the predictions and the lower one
    shows the standard deviation of the predictive distribution at each point.
    """

    if ax is None:
        _, ax = plt.subplots(2, 1, gridspec_kw={"height_ratios": [3, 1]})

    # Some datasets may not be labeled.
    if y is not None:
        scatter_data(X, y, label="Points", color="blue", ax=ax[0])

    # Plot predictive mean with confidence interval using predictive sqrt
    plot_prediction(
        X,
        predictive_mean.flatten(),
        predictive_std.flatten(),
        label="VIP Predictive Mean",
        mean_color="#029386",
        std_color="#cfe6fc",
        ax=ax[0],
    )

    # Plot each of the used samples from the Gaussian Mixture
    for i, pred in enumerate(means):
        plot_prediction(
            X,
            pred.flatten(),
            mean_color="#029386",
            alpha=0.3,
            ax=ax[0],
            label="VIP Posterior Sample" if i == 0 else None,
        )

    plot_standard_deviation(
        X,
        predictive_std.flatten(),
        color="#cfe6fc",
        alpha=0.8,
        label="VIP Prediction Standard Deviation",
        ax=ax[1],
    )
    if prior_samples is not None:
        plot_prior_samples(X, prior_samples, ax[0])

    ax[0].legend()
    ax[1].legend()


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
        _, ax = plt.subplots()

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


def plot_standard_deviation(X, std, color=None, alpha=1.0, label=None, ax=None):
    if ax is None:
        fig, ax = plt.subplots()

    sort = np.argsort(X)
    X = X[sort]
    std = std[sort]
    ax.fill_between(X, np.zeros_like(std), std, color=color, label=label, alpha=alpha)

    return ax


def plot_prior_over_layers(X, prior_samples, n=2):

    X = X.flatten()
    n_layers = prior_samples.shape[0]
    if n_layers == 1:
        _, ax = plt.subplots(figsize=(5, 15))
        plot_prior_samples(X.flatten(), prior_samples[0], ax)
        ax.set_title("Layer")
    else:
        _, ax = plt.subplots(n, n_layers // n, figsize=(5, 15))

        for i in range(n_layers):
            plot_prior_samples(X.flatten(), prior_samples[i], ax[i // n][i % n])

            ax[i // n][i % n].set_title("Layer {}".format(i + 1))
    plt.suptitle("Prior Samples")
    plt.show()


def plot_prior_samples(X, prior_samples, ax):

    sort = np.argsort(X)

    for i in range(prior_samples.shape[0]):
        plot_prediction(
            X[sort],
            prior_samples[i][sort].flatten(),
            label="Prior samples" if i == 0 else "",
            mean_color="red",
            alpha=0.1,
            ax=ax,
        )
    ax.legend()


def learning_curve(df, df_val, test_metrics_names, num_metrics, args):
    fig = plt.figure(figsize=(20, 10))

    ax3 = fig.add_subplot(2, 2, 2)
    ax4 = fig.add_subplot(2, 2, 4)

    loss = df[["LOSS"]].to_numpy().flatten()
    ax3.plot(loss, label="Training loss")
    ax3.legend()
    ax3.set_title("Loss evolution")
    ax4.plot(
        np.arange(loss.shape[0] // 5, loss.shape[0]),
        loss[loss.shape[0] // 5 :],
        label="Training loss",
    )
    ax4.legend()
    ax4.set_title("Loss evolution in last half of epochs")

    for i, m in enumerate(test_metrics_names[1:]):
        ax = fig.add_subplot(num_metrics - 1, 2, 2 * i + 1)
        ax.plot(df[[m]].to_numpy(), label="Training {}".format(m))
        ax.plot(df_val[[m]].to_numpy(), label="Validation {}".format(m))
        ymin, ymax = ax.get_ylim()
        d = (ymax - ymin) / 10
        ax.vlines(
            np.argmin(df[[m]].to_numpy()),
            np.min(df[[m]].to_numpy()) - d,
            np.min(df[[m]].to_numpy()) + d,
            color="black",
            label="Minimum value",
        )
        ax.vlines(
            np.argmin(df_val[[m]].to_numpy()),
            np.min(df_val[[m]].to_numpy()) - d,
            np.min(df_val[[m]].to_numpy()) + d,
            color="black",
        )
        if m == "RMSE":
            ax.set_yscale("log")
        ax.legend()
        ax.set_title("{} evolution".format(m))

    plt.savefig("plots/" + create_file_name(args) + ".png")
    # open file for writing

    if args.show:
        plt.show()
