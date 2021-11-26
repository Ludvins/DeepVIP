import numpy as np
from tqdm import tqdm
from utils import *
import tensorflow as tf


def train(model, train_dataset, optimizer, epochs=20000, batch_size=32):

    train_dataset = train_dataset.batch(batch_size)

    with tqdm(range(epochs), unit="epoch", miniters=10) as tepoch:
        for epoch in tepoch:
            tepoch.set_description("Training ")
            # Iterate over the batches of the dataset.
            for x_batch, y_batch in train_dataset:

                if model.dtype != x_batch.dtype:
                    x_batch = tf.cast(x_batch, model.dtype)
                if model.dtype != y_batch.dtype:
                    y_batch = tf.cast(y_batch, model.dtype)

                with tf.GradientTape() as tape:
                    # Forward pass
                    mean_pred, std_pred = model(x_batch)

                    # Compute loss function
                    loss = model.nelbo(
                        x_batch,
                        y_batch,
                    )
                    # Compute gradients
                    gradients = tape.gradient(loss, model.trainable_variables)

                    # Update weights
                    optimizer.apply_gradients(
                        zip(gradients, model.trainable_variables)
                    )

                    model.update_metrics(y_batch, mean_pred, std_pred, loss)

            if epoch % tepoch.miniters == 0:
                tepoch.set_postfix(
                    {
                        "nelbo_train": "{:2f}".format(model.metrics["nelbo"]),
                        "rmse_train": "{:2f}".format(model.metrics["rmse"]),
                        "nll_train": "{:2f}".format(model.metrics["nll"]),
                    }
                )
            model.reset_metrics()


def predict(model, dataset, batch_size=32):

    dataset = dataset.batch(batch_size)

    # Generate variables and operar)
    means, stds = [], []
    for data in dataset:
        try:
            batch_x, _ = data
        except:
            batch_x = data

        batch_means, batch_stds = model(batch_x)
        means.append(batch_means)
        stds.append(batch_stds)

    means = np.concatenate(means, axis=1)
    stds = np.concatenate(stds, axis=1)

    return means, stds


def predict_prior_samples(model, generator, device=None):

    batch_size = generator.batch_size
    length_dataset = len(generator.dataset)

    assert batch_size <= length_dataset

    # Generate variables and operations for the minimizer and initialize variables
    train_per_epoch = len(generator)
    prior = []
    for idx, data in enumerate(generator):
        try:
            batch_x, _ = data
        except:
            batch_x = data
        prior_samples = model.predict_prior_samples(batch_x.to(device))

        prior.append(prior_samples.detach().numpy())

    prior = np.concatenate(prior)

    return prior
