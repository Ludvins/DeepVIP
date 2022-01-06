import numpy as np
from .metrics import Metrics
from tqdm import tqdm
import tensorflow as tf


def fit(
    model,
    training_generator,
    optimizer,
    epochs=2000,
):
    """
    Trains the given model using the arguments provided.

    Arguments
    ---------
    model : torch.nn.Module
            Torch model to train.
    training_generator : iterable
                         Must return batches of pairs corresponding to the
                         given inputs and target values.
    epochs : int
             Number of epochs to train de model.
    device : torch device
             Device in which to perform all computations.
    """
    for _ in range(epochs):
        # Mini-batch training
        for inputs, targets in training_generator:

            if model.dtype != inputs.dtype:
                inputs = tf.cast(inputs, model.dtype)
            if model.dtype != targets.dtype:
                targets = tf.cast(targets, model.dtype)

            with tf.GradientTape() as tape:
                # Forward pass
                mean_pred, std_pred = model(inputs)

                # Compute loss function
                loss = model.nelbo(
                    inputs,
                    targets,
                )
                # Compute gradients
                gradients = tape.gradient(loss, model.trainable_variables)

                # Update weights
                optimizer.apply_gradients(
                    zip(gradients, model.trainable_variables)
                )


def score(model, generator, device=None):
    """
    Evaluates the given model using the arguments provided.

    Arguments
    ---------
    model : torch.nn.Module
            Torch model to train.
    generator : iterable
                Must return batches of pairs corresponding to the
                given inputs and target values.
    device : torch device
             Device in which to perform all computations.

    Returns
    -------
    metrics : dictionary
              Contains pairs of (metric, value) averaged over the number of
              batches.
    """
    # Initialize metrics
    metrics = Metrics()
    # Batches evaluation
    for inputs, targets in generator:

        if model.dtype != inputs.dtype:
            inputs = tf.cast(inputs, model.dtype)
        if model.dtype != targets.dtype:
            inputs = tf.cast(targets, model.dtype)

        mean_pred, std_pred = model(inputs)

        # Compute loss function
        loss = model.nelbo(
            inputs,
            targets,
        )

        metrics.update(targets, loss, mean_pred, std_pred, light=False)
    # Return metrics as a dictionary
    return metrics.get_dict()


def predict(model, generator, device=None):

    with torch.no_grad():
        # Generate variables and operar)
        means, vars = [], []
        for idx, data in enumerate(generator):
            try:
                batch_x, _ = data
            except:
                batch_x = data
            batch_means, batch_vars = model(batch_x.to(device))
            means.append(batch_means.detach().cpu().numpy())
            vars.append(batch_vars.detach().cpu().numpy())

        means = np.concatenate(means, axis=1)
        vars = np.concatenate(vars, axis=1)

    return means, vars


def fit_with_metrics(
    model,
    training_generator,
    optimizer,
    val_generator=None,
    epochs=2000,
    verbose=1,
):

    # Array storing metrics during training
    metrics = Metrics()
    if val_generator is not None:
        metrics_val = Metrics()

    history = []
    history_val = []

    # TQDM update interval
    miniters = 10

    if verbose == 1:
        # initialize TQDM bar
        tepoch = tqdm(range(epochs), unit="epoch", miniters=miniters)
        tepoch.set_description("Training ")
    else:
        tepoch = range(epochs)

    for epoch in tepoch:
        # Mini-batch training
        model.train_mode()
        for inputs, targets in training_generator:
            if model.dtype != inputs.dtype:
                inputs = tf.cast(inputs, model.dtype)
            if model.dtype != targets.dtype:
                targets = tf.cast(targets, model.dtype)

            with tf.GradientTape() as tape:
                # Forward pass
                mean_pred, std_pred = model(inputs)

                # Compute loss function
                loss = model.nelbo(
                    inputs,
                    targets,
                )
                # Compute gradients
                gradients = tape.gradient(loss, model.trainable_variables)
                # Update weights
                optimizer.apply_gradients(
                    zip(gradients, model.trainable_variables)
                )

            metrics.update(
                targets * model.y_std + model.y_mean,
                loss,
                mean_pred,
                std_pred,
            )

        # Store history of metrics
        metrics_dict = metrics.get_dict()
        history.append(metrics_dict)
        # Reset current metrics for next epochs or validation
        metrics.reset()

        val_postfix = {}
        model.eval_mode()
        if val_generator is not None:

            for inputs, targets in val_generator:

                if model.dtype != inputs.dtype:
                    inputs = tf.cast(inputs, model.dtype)
                if model.dtype != targets.dtype:
                    targets = tf.cast(targets, model.dtype)

                mean_pred, std_pred = model(inputs)

                metrics_val.update(
                    targets,
                    0,
                    mean_pred,
                    std_pred,
                )

            metrics_val_dict = metrics_val.get_dict()
            metrics_val.reset()
            history_val.append(metrics_val_dict)

            # Handle Validation metrics in TQDM
            if verbose == 1:
                val_postfix = {
                    "rmse_val": "{0:.2f}".format(metrics_val_dict["RMSE"]),
                    "nll_val": "{0:.2f}".format(metrics_val_dict["NLL"]),
                }

        # Show metrics in TQDM
        if verbose == 1 and epoch % miniters == 0:
            tepoch.set_postfix(
                {
                    **{
                        "loss_train": "{0:.2f}".format(metrics_dict["LOSS"]),
                        "rmse_train": "{0:.2f}".format(metrics_dict["RMSE"]),
                        "nll_train": "{0:.2f}".format(metrics_dict["NLL"]),
                    },
                    **val_postfix,
                }
            )

    if val_generator is None:
        return history
    return history, history_val
