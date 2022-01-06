import tensorflow as tf


class DVIP_Base(tf.Module):
    def __init__(
        self,
        likelihood,
        layers,
        num_data,
        num_samples=1,
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
                     Indicates the likelihood distribution of the data
        layers : array of Layer
                 Contains the different Variational Implicit Process layers
                 that make up this model.
        num_data : int
                   Ammount of data samples
        y_mean : float or array-like
                 Original value of the normalized labels
        y_std : float or array-like
                Original standar deviation of the normalized labels
        dtype : data-type
                The dtype of the layer's computations and weights.
                Refer to tf.keras.Model for more information.
        **kwargs : dict, optional
                   Extra arguments to `Model`.
                   Refer to tf.keras.Model for more information.
        """
        super().__init__(name="DVIP_Base", **kwargs)
        self.num_data = num_data
        self.num_samples = num_samples
        self.y_mean = y_mean
        self.y_std = y_std

        self.likelihood = likelihood
        self.vip_layers = layers

        self.dtype = dtype
        # Metric trackers
        self.loss_tracker = tf.keras.metrics.Mean(name="nelbo")

    def train_mode(self):
        self.training = True
        for layer in self.vip_layers:
            layer.train_mode()

    def eval_mode(self):
        self.training = False
        for layer in self.vip_layers:
            layer.eval_mode()

    def __call__(self, inputs):
        """
        Computes the prediction of the model. Calls `predict_y`.

        Parameters
        ----------
        inputs : tf.tensor of shape (num_data, data_dim)
                 Contains the features whose labels are to be predicted
        Returns
        -------
        mean : tf.tensor of shape (num_data, output_dim)
               Contains the mean value of the predictive distribution
        std : tf.tensor of shape (num_data, output_dim)
              Contains the std of the predictive distribution
        """
        mean, var = self.predict_y(inputs, num_samples=self.num_samples)

        return mean * self.y_std + self.y_mean, tf.math.sqrt(var) * self.y_std

    def propagate(self, X, num_samples=1, full_cov=False):
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
        # Define arrays
        Fs, Fmeans, Fvars = [], [], []
        # First input corresponds to the original one
        F = tf.tile(tf.expand_dims(X, 0), [num_samples, 1, 1])
        for layer in self.vip_layers:
            F, Fmean, Fvar = layer.sample_from_conditional(
                F, full_cov=full_cov
            )
            # Store values
            Fs.append(F)
            Fmeans.append(Fmean)
            Fvars.append(Fvar)

        # Return arrays
        return Fs, Fmeans, Fvars

    def predict_f(self, predict_at, num_samples, full_cov=False):
        """
        Returns the predicted mean and variance at the last layer.

        Parameters
        ----------
        predict_at : tf.tensor of shape (num_data, data_dim)
                     Contains the input features.
        full_cov : boolean
                   Whether to use the full covariance matrix or just
                   the diagonal values.
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
            predict_at,
            num_samples=num_samples,
            full_cov=full_cov,
        )
        return Fmeans[-1], Fvars[-1]

    def predict_y(self, predict_at, num_samples, full_cov=False):
        """
        Computes the predicted labels for the given input.

        Parameters
        ----------
        predict_at : tf.tensor of shape (num_data, data_dim)
                     Contains the input features.
        full_cov : boolean
                   Whether to use the full covariance matrix or just
                   the diagonal values.
        Returns
        -------
        The predicted labels using the model's likelihood
        and the predicted mean and standard deviation.
        """
        mean, var = self.predict_f(
            predict_at, num_samples=num_samples, full_cov=full_cov
        )
        return self.likelihood.predict_mean_and_var(mean, var)

    def predict_log_density(self, data):
        Fmean, Fvar = self.predict_f(data[0], full_cov=False)
        l = self.likelihood.predict_logdensity(Fmean, Fvar, data[1])
        log_num_samples = tf.math.log(tf.cast(self.num_samples, self.dtype))
        return tf.reduce_logsumexp(l - log_num_samples, axis=0)

    def expected_data_log_likelihood(self, X, Y):
        """
        Compute expectations of the data log likelihood under the variational
        distribution

        Parameters
        ----------
        X : tf.tensor of shape (num_samples, input_dim)
            Contains input features

        Y : tf.tensor of shape (num_samples, output_dim)
            Contains input labels

        Returns
        -------
        var_exp : tf.tensor of shape (1)
                  Contains the variational expectation

        """
        F_mean, F_var = self.predict_f(
            X, num_samples=self.num_samples, full_cov=False
        )
        # Shape [S, N, D]
        var_exp = self.likelihood.variational_expectations(F_mean, F_var, Y)
        # Shape [N, D]
        return tf.reduce_mean(var_exp, 0)

    def nelbo(self, inputs, outputs, iteration=None):
        """
        Computes the evidence lower bound.

        Parameters
        ----------
        inputs : tf.tensor of shape (num_samples, input_dim)
                 Contains input features

        outputs : tf.tensor of shape (num_samples, output_dim)
                  Contains input labels

        Returns
        -------
        nelbo : tf.tensor
                Negative evidence lower bound
        """
        X, Y = inputs, outputs
        likelihood = tf.reduce_sum(self.expected_data_log_likelihood(X, Y))
        # scale loss term corresponding to minibatch size
        scale = tf.cast(self.num_data, self.dtype)
        scale /= tf.cast(tf.shape(X)[0], self.dtype)
        # Compute KL term
        #
        KL = tf.reduce_sum([layer.KL() for layer in self.vip_layers])
        return -scale * likelihood + KL
