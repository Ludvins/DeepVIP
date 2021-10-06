# Based on GPflow's likelihood

import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np


class Gaussian(tf.keras.layers.Layer):
    def __init__(self, variance=1.0, dtype=tf.float64, **kwargs):
        """"""
        super().__init__(dtype=dtype)

        transform = tfp.bijectors.Softplus()
        self.variance = tfp.util.TransformedVariable(
            variance, bijector=transform, dtype=self.dtype, name="lik_log_variance"
        )

        self.rmse_metric = tf.keras.metrics.RootMeanSquaredError(name="rmse")

        self.nll_metric = tf.keras.metrics.Mean(name="nll")

        self.build(0)

    @property
    def metrics(self):
        return [self.rmse_metric, self.nll_metric]

    @tf.autograph.experimental.do_not_convert
    def update_metrics(self, y, mean_pred, std_pred):
        if tf.shape(mean_pred).shape == 3:
            predictions = tf.reduce_mean(mean_pred, 0)
        else:
            predictions = mean_pred
        self.rmse_metric.update_state(y, predictions)

        S = tf.cast(tf.shape(mean_pred)[0], dtype=self.dtype)
        normal = tfp.distributions.Normal(loc=mean_pred, scale=std_pred)
        logpdf = normal.log_prob(y)
        nll = tf.math.reduce_logsumexp(logpdf, 0) - tf.math.log(S)
        nll = -tf.reduce_mean(nll)

        self.nll_metric.update_state(nll)

    def logdensity(self, x, mu, var):
        return -0.5 * (np.log(2 * np.pi) + tf.math.log(var) + tf.square(mu - x) / var)

    def logp(self, F, Y):
        return self.logdensity(Y, F, self.variance)

    def conditional_mean(self, F):
        return tf.identity(F)

    def conditional_variance(self, F):
        return tf.fill(tf.shape(F), tf.squeeze(self.variance))

    def predict_mean_and_var(self, Fmu, Fvar):
        return tf.identity(Fmu), Fvar + self.variance

    def predict_logdensity(self, Fmu, Fvar, Y):
        return self.logdensity(Y, Fmu, Fvar + self.variance)

    def variational_expectations(self, Fmu, Fvar, Y):
        return (
            -0.5 * np.log(2 * np.pi)
            - 0.5 * tf.math.log(self.variance)
            - 0.5 * (tf.square(Y - Fmu) + Fvar) / self.variance
        )