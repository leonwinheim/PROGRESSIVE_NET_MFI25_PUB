"""
MCMC.py

This module implements a Markov Chain Monte Carlo (MCMC) based Bayes Neural Network (BNN) model.
Numpyro is used as a backend for probabilistic programming.

"""
import jax
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS
import matplotlib.pyplot as plt

class MCMC_NET:
    """
    This implementation only supports a single hidden layer neural network.
    """
    def __init__(self, in_features, hidden_units, out_features, likelihood_noise=None):
        self.in_features = in_features
        self.hidden_units = hidden_units
        self.out_features = out_features
        self.likelihood_noise = likelihood_noise

    def model(self,x, y=None):
        if self.likelihood_noise is None:
            in_features = self.in_features
            hidden_units = self.hidden_units
            out_features = self.out_features

            # Hidden layer weights and biases
            w1 = numpyro.sample("w1", dist.Normal(0., 1.).expand([in_features, hidden_units]))
            b1 = numpyro.sample("b1", dist.Normal(0., 1.).expand([hidden_units]))

            # Output layer weights and biases
            w2 = numpyro.sample("w2", dist.Normal(0., 1.).expand([hidden_units, out_features]))
            b2 = numpyro.sample("b2", dist.Normal(0., 1.).expand([out_features]))

            # Observation noise
            sigma = numpyro.sample("sigma", dist.Uniform(0., 10.))

            # Forward pass
            x = x.reshape(-1, in_features)  # ensure 2D
            hidden = jax.nn.relu(jnp.dot(x, w1) + b1)
            mean = jnp.dot(hidden, w2) + b2
            mean = mean.squeeze(-1)  # shape: (batch,)

            # Likelihood
            with numpyro.plate("data", x.shape[0]):
                numpyro.sample("obs", dist.Normal(mean, sigma), obs=y)
        else:
            in_features = self.in_features
            hidden_units = self.hidden_units
            out_features = self.out_features

            # Hidden layer weights and biases
            w1 = numpyro.sample("w1", dist.Normal(0., 1.).expand([in_features, hidden_units]))
            b1 = numpyro.sample("b1", dist.Normal(0., 1.).expand([hidden_units]))

            # Output layer weights and biases
            w2 = numpyro.sample("w2", dist.Normal(0., 1.).expand([hidden_units, out_features]))
            b2 = numpyro.sample("b2", dist.Normal(0., 1.).expand([out_features]))

            # Observation noise
            sigma = jnp.sqrt(self.likelihood_noise)

            # Forward pass
            x = x.reshape(-1, in_features)  # ensure 2D
            hidden = jax.nn.relu(jnp.dot(x, w1) + b1)
            mean = jnp.dot(hidden, w2) + b2
            mean = mean.squeeze(-1)  # shape: (batch,)

            # Likelihood
            with numpyro.plate("data", x.shape[0]):
                numpyro.sample("obs", dist.Normal(mean, sigma), obs=y)