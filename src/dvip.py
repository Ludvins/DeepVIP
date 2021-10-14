import tensorflow as tf


class DVIP_Base(tf.keras.Model):
    def __init__(
        self,
        likelihood,
        layers,
        num_data,
        y_mean=0.0,
        y_std=1.0,
        dtype=tf.float64,
        **kwargs
    ):
        """
        Defines a Base class for Deep Variational Implicit Processes as a
        particular Keras model.

        Parameters
        ----------
        likelihood : Likelihood


        layers : array of Layer
                 Contains the different Variational Implicit Process layers
                 that make up this model.
        num_data : int
                   Ammount of data samples
        y_mean : float
                 Original value of the normalized labels
        y_std : float
                Original standar deviation of the normalized labels
        dtype : data-type
                The dtype of the layer's computations and weights.
                Refer to tf.keras.Model for more information.
        **kwargs : dict, optional
                   Extra arguments to `Model`.
                   Refer to tf.keras.Model for more information.
        """
        super().__init__(name="DVIP_Base", dtype=dtype, **kwargs)
        self.num_data = num_data
        self.y_mean = y_mean
        self.y_std = y_std

        self.likelihood = likelihood
        self.vip_layers = layers

        # Metric trackers
        self.loss_tracker = tf.keras.metrics.Mean(name="nelbo")

    @tf.function
    def train_step(self, data):
        """
        Defines the training step for the DVIP model.

        Parameters
        ----------
        data : tuple of shape
               ([num_samples, data_dim], [num_samples, labels_dim])
               Contains features and labels of the training set.

        Returns
        -------
        metrics : dictionary
                  Contains the resulting metrics of the training step.
        """
        # Split data in features (x) and labels (y)
        x, y = data

        # Ensure types are compatible
        if eval("tf." + self.dtype) != x.dtype:
            x = tf.cast(x, self.dtype)
        if eval("tf." + self.dtype) != y.dtype:
            y = tf.cast(y, self.dtype)

        with tf.GradientTape() as tape:
            # Forward pass
            mean_pred, var_pred = self(x, training=True)

            # Compute loss function
            loss = self.nelbo(x, y)

            # Get trainable variables from the model itself
            trainable_vars = self.trainable_variables

            # Compute gradients
            gradients = tape.gradient(loss, trainable_vars)

        # Update weights
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))

        # Compute metrics
        self.loss_tracker.update_state(loss)

        # Update metrics
        self.likelihood.update_metrics(
            y * self.y_std + self.y_mean, mean_pred, var_pred
        )

        return {m.name: m.result() for m in self.metrics}

    @tf.function
    def test_step(self, data):
        """
        Defines the test step for the DVIP model.

        Parameters
        ----------
        data : tuple of shape
               ([num_samples, data_dim], [num_samples, labels_dim])
               Contains features and labels of the training set.

        Returns
        -------
        metrics : dictionary
                  Contains the resulting metrics of the training step.
        """
        # Unpack the data
        x, y = data

        # Ensure types are compatible
        if eval("tf." + self.dtype) != x.dtype:
            x = tf.cast(x, self.dtype)
        if eval("tf." + self.dtype) != y.dtype:
            y = tf.cast(y, self.dtype)

        # Compute predictions
        mean_pred, var_pred = self(x, training=False)

        # Compute the loss
        loss = self.nelbo(x, (y - self.y_mean) / self.y_std)

        # Update the metrics
        self.loss_tracker.update_state(loss)
        self.likelihood.update_metrics(y, mean_pred, var_pred)

        # Return a dict mapping metric names to current value.
        # Note that it will include the loss (tracked in self.metrics).
        return {m.name: m.result() for m in self.metrics}

    @property
    def metrics(self):
        # We list our `Metric` objects here so that `reset_states()` can be
        # called automatically at the start of each epoch
        # or at the start of `evaluate()`.
        # If you don't implement this property, you have to call
        # `reset_states()` yourself at the time of your choosing.
        metrics = [self.loss_tracker]
        for metric in self.likelihood.metrics:
            metrics.append(metric)
        return metrics

    @tf.function
    def call(self, inputs):
        """
        Computes the prediction of the model. Calls `predict_y` and scales
        the output.

        Parameters
        ----------
        inputs : tf.tensor of shape (num_data, data_dim)
                 Contains the features whose label is to be predicted
        Returns
        -------
        mean : tf.tensor of shape (num_data, output_dim)
               Contains the mean value of the predictive distribution
        var : tf.tensor of shape (num_data, output_dim)
              Contains the variance of the predictive distribution
        """
        # Make prediction
        output_means, output_vars = self.predict_y(inputs)

        # Scale output
        mean = output_means * self.y_std + self.y_mean
        var = tf.math.sqrt(output_vars) * self.y_std
        return mean, var

    @tf.function
    def propagate(self, X, full_cov=False, bootstrap_replicates=1):
        """
        Propagates the input trough the layer, using the output of the previous
        as the input for the next one.

        Parameters
        ----------
        X : tf.tensor of shape (num_data, data_dim)
            Contains the input features.
        full_cov : boolean
                   Whether to use the full covariance matrix or just
                   the diagonal values.
        bootstrap_replicates : int
                               Number of bootstrap replicates to use.

        Returns
        -------
        Fs : tf.tensor of shape (num_layers, num_data, data_dim)
             Contains the propagation made in each layer
        Fmeans : tf.tensor of shape (num_layers, num_data, output_dim)
                 Contains the mean value of the predictions at
                 each layer.
        Fvars : tf.tensor of shape (num_layers, num_data, output_dim)
                Contains the standard deviation of the predictions at
                each layer.
        """
        # Create bootstrap replicates
        sX = tf.tile(tf.expand_dims(X, 0), [bootstrap_replicates, 1, 1])
        # Define arrays
        Fs, Fmeans, Fvars = [], [], []

        # First input corresponds to the original one
        F = sX

        for layer in self.vip_layers:
            F, Fmean, Fvar = layer.sample_from_conditional(F,
                                                           full_cov=full_cov)
            # Store values
            Fs.append(F)
            Fmeans.append(Fmean)
            Fvars.append(Fvar)

        # Return arrays
        return Fs, Fmeans, Fvars

    def predict_f(self, predict_at, bootstrap_replicates=1, full_cov=False):
        """
        Returns the predicted mean and variance at the last layer.

        Parameters
        ----------
        predict_at : tf.tensor of shape (num_data, data_dim)
                     Contains the input features.
        full_cov : boolean
                   Whether to use the full covariance matrix or just
                   the diagonal values.
        bootstrap_replicates : int
                               Number of bootstrap replicates to use.

        Returns
        -------
        Fmeans : tf.tensor of shape (num_data, output_dim)
                 Contains the mean value of the predictions at
                 the last layer.
        Fvars : tf.tensor of shape (num_data, output_dim)
                Contains the standard deviation of the predictions at
                the last layer.

        """
        _, Fmeans, Fvars = self.propagate(
            predict_at, full_cov=full_cov,
            bootstrap_replicates=bootstrap_replicates
        )
        return Fmeans[-1], Fvars[-1]

    def predict_all_layers(self, predict_at, bootstrap_replicates,
                           full_cov=False):
        """
        Propagates the input and returns all the predicted data.

        Parameters
        ----------
        predict_at : tf.tensor of shape (num_data, data_dim)
                     Contains the input features.
        full_cov : boolean
                   Whether to use the full covariance matrix or just
                   the diagonal values.
        bootstrap_replicates : int
                               Number of bootstrap replicates to use.

        Returns
        -------
        Fs : tf.tensor of shape (num_layers, num_data, data_dim)
             Contains the propagation made in each layer
        Fmeans : tf.tensor of shape (num_layers, num_data, output_dim)
                 Contains the mean value of the predictions at
                 each layer.
        Fvars : tf.tensor of shape (num_layers, num_data, output_dim)
                Contains the standard deviation of the predictions at
                each layer.
        """

        return self.propagate(predict_at, full_cov=full_cov,
                              bootstrap_replicates=bootstrap_replicates)

    def predict_y(self, predict_at, bootstrap_replicates=1):
        """
        Computes the predicted labels for the given input.

        Parameters
        ----------
        predict_at : tf.tensor of shape (num_data, data_dim)
                     Contains the input features.
        bootstrap_replicates : int
                               Number of bootstrap replicates to use.

        Returns
        -------
        The predicted labels using the model's likelihood
        and the predicted mean and variance.
        """

        Fmean, Fvar = self.predict_f(
            predict_at, bootstrap_replicates=bootstrap_replicates,
            full_cov=False
        )
        return self.likelihood.predict_mean_and_var(Fmean, Fvar)

    def predict_log_density(self, data, bootstrap_replicates):
        Fmean, Fvar = self.predict_f(data[0], bootstrap_replicates=bootstrap_replicates, full_cov=False)
        l = self.likelihood.predict_density(Fmean, Fvar, data[1])
        log_num_samples = tf.math.log(tf.cast(self.num_samples, self.dtype))
        return tf.reduce_logsumexp(l - log_num_samples, axis=0)

    @tf.function
    def expected_data_log_likelihood(self, X, Y):
        """
        Compute expectations of the data log likelihood under the variational
        distribution with MC samples
        """
        F_mean, F_var = self.predict_f(X,
                                       full_cov=False)
        # Shape [S, N, D]
        var_exp = self.likelihood.variational_expectations(F_mean, F_var, Y)
        # Shape [N, D]
        return tf.reduce_mean(var_exp, 0)

    @tf.function
    def nelbo(self, inputs, outputs):
        """
        Computes the evidence lower bound.

        Parameters
        ----------
        inputs : tf.tensor of shape
                 Contains input features

        outputs : tf.tensor of shape
                  Contains input labels

        Returns
        -------
        nelbo : tf.tensor of shape (1)
                Negative evidence lower bound
        """
        X, Y = inputs, outputs

        likelihood = tf.reduce_sum(self.expected_data_log_likelihood(X, Y))
        # scale loss term corresponding to minibatch size
        scale = tf.cast(self.num_data, self.dtype)
        scale /= tf.cast(tf.shape(X)[0], self.dtype)
        # Compute KL term
        KL = tf.reduce_sum([layer.KL() for layer in self.vip_layers])
        return -scale * likelihood + KL

    def print_variables(self):
        for variable in self.trainable_variables:
            print(variable)
