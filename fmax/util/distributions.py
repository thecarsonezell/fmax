import pymc3 as pm
import numpy as np

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
  def __init__(self, alpha, scale, *args, **kwargs):
    super().__init__(*args, **kwargs)
    self.alpha = alpha
    self.scale = scale


  def logp(self, x):
    scaled_x = x / self.scale
    logp = pm.math.log(self.alpha) - pm.math.log(self.scale) - (self.alpha + 1) * pm.math.log(scaled_x) - (scaled_x ** -self.alpha)
    return pm.math.where(scaled_x > 0, logp, -np.inf)


  def logcdf(self, x):
    scaled_x = x / self.scale
    logcdf = - (scaled_x ** -self.alpha)
    return logcdf
