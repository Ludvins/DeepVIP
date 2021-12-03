import numpy as np
import matplotlib.pyplot as plt


def build_plot_name(vip_layers, bnn_structure, input_dim, output_dim, epochs,
                    n_samples, dataset, name_flag, genf, num_inducing,
                    **kwargs):
    # Create title name
    dims = np.concatenate(
        ([input_dim], np.ones(vip_layers, dtype=int) * [output_dim]))

    dims_name = "-".join(map(str, dims))
    model_name = genf
    if genf == "BNN":
        model_name += " " + "-".join(map(str, bnn_structure))
    elif genf == "GPI":
        model_name += " " + str(num_inducing)

    path = "plots/{}_{}_layers={}_bnn={}_epochs={}_batchsize={}".format(
        dataset, name_flag, dims_name, model_name, epochs, n_samples)
    title = "Layers: {}({}) {}".format(vip_layers, dims_name, model_name)
    return title, path


def plot_train_test(train_pred,
                    test_pred,
                    X_train,
                    y_train,
                    X_test,
                    y_test=None,
                    train_prior_samples=None,
                    test_prior_samples=None,
                    title=None,
                    path=None,
                    show=True):
    mean_train, std_train = train_pred
    mean_test, std_test = test_pred

    _, ax = plt.subplots(2,
                         2,
                         gridspec_kw={"height_ratios": [3, 1]},
                         figsize=(20, 10))

    plt.suptitle(title)

    plot_results(
        X=X_train.flatten(),
        mean=mean_train,
        std=std_train,
        y=y_train,
        prior_samples=train_prior_samples[-1],
        ax=ax.T[0],
    )

    if y_test is not None:
        y_test.flatten()

    plot_results(
        X=X_test.flatten(),
        mean=mean_test,
        std=std_test,
        y=y_test,
        prior_samples=test_prior_samples[-1],
        ax=ax.T[1],
    )

    ax[0][0].set_title("Training results")
    ax[0][1].set_title("Test results")
    plt.savefig(path + ".svg", format="svg")
    plt.savefig(path + ".png", format="png")
    if show:
        plt.show()
    plt.close()


def plot_results(X, mean, std, y=None, prior_samples=None, ax=None):
    if ax is None:
        _, ax = plt.subplots(2, 1, gridspec_kw={"height_ratios": [3, 1]})

    if y is not None:
        scatter_data(X, y, label="Points", color="blue", ax=ax[0])

    prediction_mean = np.mean(mean, axis=0)
    prediction_std = np.mean(std, axis=0)
    plot_prediction(
        X,
        prediction_mean.flatten(),
        prediction_std.flatten(),
        label="VIP Predictive Mean",
        mean_color="#029386",
        std_color="#cfe6fc",
        ax=ax[0],
    )

    for i, pred in enumerate(mean):
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
        prediction_std.flatten(),
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
        fig, ax = plt.subplots()

    sort = np.argsort(X)
    X = X[sort]
    mean = mean[sort]

    ax.plot(X, mean, color=mean_color, alpha=alpha, label=label)
    if std is not None:
        std = std[sort]
        ax.fill_between(X,
                        mean - 2 * std,
                        mean + 2 * std,
                        color=std_color,
                        alpha=alpha / 2)

    return ax


def plot_standard_deviation(X,
                            std,
                            color=None,
                            alpha=1.0,
                            label=None,
                            ax=None):
    if ax is None:
        fig, ax = plt.subplots()

    sort = np.argsort(X)
    X = X[sort]
    std = std[sort]
    ax.fill_between(X,
                    np.zeros_like(std),
                    std,
                    color=color,
                    label=label,
                    alpha=alpha)

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
            plot_prior_samples(X.flatten(), prior_samples[i],
                               ax[i // n][i % n])

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
