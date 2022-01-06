# Based on GPflow's likelihood

import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np


class Likelihood(tf.keras.layers.Layer):
    def __init__(self, dtype=tf.float64, **kwargs):
        """"""
        super().__init__(dtype=dtype)
        self.build(0)

    def logdensity(self, x, mu, var):
        raise NotImplementedError

    def logp(self, F, Y):
        raise NotImplementedError

    def conditional_mean(self, F):
        raise NotImplementedError

    def conditional_variance(self, F):
        raise NotImplementedError

    def predict_mean_and_var(self, Fmu, Fvar):
        raise NotImplementedError

    def predict_logdensity(self, Fmu, Fvar, Y):
        raise NotImplementedError

    def variational_expectations(self, Fmu, Fvar, Y):
        raise NotImplementedError


class Gaussian(Likelihood):
    def __init__(self, log_variance=-5.0, dtype=tf.float64, **kwargs):
        """"""

        self.log_variance = tf.Variable(
            initial_value=log_variance, dtype=dtype, name="lik_log_variance"
        )

        super().__init__(dtype=dtype, **kwargs)

    def logdensity(self, x, mu, var):
        return -0.5 * (
            np.log(2 * np.pi) + tf.math.log(var) + tf.square(mu - x) / var
        )

    def logp(self, F, Y):
        return self.logdensity(Y, F, tf.math.exp(self.log_variance))

    def conditional_mean(self, F):
        return tf.identity(F)

    def conditional_variance(self, F):
        return tf.fill(tf.shape(F), tf.squeeze(tf.math.exp(self.log_variance)))

    def predict_mean_and_var(self, Fmu, Fvar):
        return tf.identity(Fmu), Fvar + tf.math.exp(self.log_variance)

    def predict_logdensity(self, Fmu, Fvar, Y):
        return self.logdensity(Y, Fmu, Fvar + tf.math.exp(self.log_variance))

    def variational_expectations(self, Fmu, Fvar, Y):

        # Get variance
        variance = tf.math.exp(self.log_variance)
        # Compute variational expectations
        var_exp = (
            -0.5 * np.log(2 * np.pi)
            - 0.5 * self.log_variance
            - 0.5 * (tf.square(Fmu - Y) + Fvar) / variance
        )
        return tf.reduce_sum(var_exp, -1)
