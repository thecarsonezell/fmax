import pymc3 as pm
import numpy as np
import theano.tensor as tt


class MinGumbel(pm.Continuous):
  def __init__(self, mu, beta):
    self.mu = mu
    self.beta = beta
  
  def logp(self, x):
    y = (x - self.mu) / self.beta
    return y - pm.math.exp(y) - pm.math.log(self.beta)
  
  def logcdf(self, x):
    y = (x - self.mu) / self.beta
    return pm.math.log(1-pm.math.exp(-pm.math.exp(y)))


class MaxWeibull(pm.Continuous):
  def __init__(self, alpha, beta):
    self.alpha = alpha
    self.beta = beta
  
  def logp(self, x):
    y = (x) / self.beta
    return pm.math.log(self.alpha) + (self.alpha - 1)*pm.math.log(-y) \
           - (-y)**self.alpha - pm.math.log(self.beta)

  def logcdf(self, x):
    y = (x) / self.beta
    return -(-y)**self.alpha
  
class Frechet(pm.Continuous):
    def __init__(self, alpha, sigma, *args, **kwargs):
        super(Frechet, self).__init__(*args, **kwargs)
        self.alpha = alpha
        self.sigma = sigma

    def logp(self, x):
        # Add a condition to handle x <= 0
        x = pm.math.switch(pm.math.gt(x, 0), x, 1e-12)

        scaled_x = x / self.sigma
        logp = pm.math.log(self.alpha) - pm.math.log(self.sigma) - (self.alpha + 1) * pm.math.log(scaled_x) - (scaled_x ** -self.alpha)
        return pm.math.switch(pm.math.gt(scaled_x, 0), logp, -np.inf)

    def logcdf(self, value):
        alpha, sigma = self.alpha, self.sigma
        scaled_value = value / sigma
        logcdf = -scaled_value ** (-alpha)

        # Debugging: Print values in logcdf method


        return logcdf


