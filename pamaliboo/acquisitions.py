"""
Copyright 2023 Bruno Guindani
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
  http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

from abc import ABC, abstractmethod
import logging
import numpy as np
from scipy.optimize import minimize
from scipy.stats import norm
from sklearn.gaussian_process import GaussianProcessRegressor as GPR

from .utils import dict_to_array


class AcquisitionFunction(ABC):
  def __init__(self, maximize_n_warmup: int, maximize_n_iter: int):
    self.logger = logging.getLogger(__name__)
    self.n_warmup = maximize_n_warmup
    self.n_iter = maximize_n_iter

  def update_state(self, gp: GPR, num_iter: int) -> None:
    self.gp = gp

  @abstractmethod
  def evaluate(self, x: np.ndarray) -> float:
    pass

  def maximize(self, bounds: dict[str: tuple[float, float]]) \
               -> tuple[np.ndarray, float]:
    self.logger.debug("Maximizing within bounds %s...", bounds)

    # Sample warmup points to evaluate the acquisition
    rng = np.random.default_rng()
    bounds_arr = dict_to_array(bounds)
    x_tries = rng.uniform(bounds_arr[:, 0], bounds_arr[:, 1],
                          size=(self.n_warmup, bounds_arr.shape[0]))
    ys = self.evaluate(x_tries)
    # Find best warmup point
    idx = ys.argmax()
    x_max = x_tries[idx]
    max_acq = ys[idx]

    # Sample initial points for minimization rounds
    x_seeds = rng.uniform(bounds_arr[:, 0], bounds_arr[:, 1],
                          size=(self.n_iter, bounds_arr.shape[0]))

    for x_try in x_seeds:
      # Find the minimum of minus the acquisition function
      res = minimize(lambda x: -self.evaluate(x), x0=x_try.reshape(1, -1),
                     bounds=bounds_arr, method=self.solver)

      if not res.success:
          continue

      # Store it if better than previous best
      if -np.squeeze(res.fun) >= max_acq:
          x_max = res.x
          max_acq = -np.squeeze(res.fun)

    x_ret = np.clip(x_max, bounds_arr[:, 0], bounds_arr[:, 1])
    self.logger.debug("Maximizer is %s, with acquisition value %f", x_ret,
                                                                    max_acq)
    return x_ret, max_acq


class UpperConfidenceBound(AcquisitionFunction):
  solver = 'L-BFGS-B'

  def __init__(self, kappa: float = 2.576, *args, **kwargs):
    self.kappa = kappa
    super().__init__(*args, **kwargs)

  def update_state(self, gp: GPR, num_iter: int) -> None:
    super().update_state(gp, num_iter)

  def evaluate(self, x: np.ndarray) -> float:
    mean, std = self.gp.predict(x, return_std=True)
    return mean + self.kappa * std


class ExpectedImprovement(AcquisitionFunction):
  solver = 'L-BFGS-B'

  def __init__(self, xi: float = 0.0, *args, **kwargs):
    self.xi = xi
    super().__init__(*args, **kwargs)

  def update_state(self, gp: GPR, num_iter: int) -> None:
    super().update_state(gp, num_iter)
    self.y_max = gp.y_train_.max()
    self.logger.debug("New EI incumbent is %f", self.y_max)

  def evaluate(self, x: np.ndarray) -> float:
    mean, std = self.gp.predict(x, return_std=True)

    a = (mean - self.y_max - self.xi)
    z = a / std
    return a * norm.cdf(z) + std * norm.pdf(z)
