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
from sklearn.base import BaseEstimator
from sklearn.gaussian_process import GaussianProcessRegressor as GPR

from .dataframe import FileDataFrame
from .utils import dict_to_array


class AcquisitionFunction(ABC):
  """
  Acquisition function to be used in Bayesian Optimization (BO) algorithms.

  BO algorithms iteratively choose the next point to evaluate by solving a
  proxy problem, which is the maximization of an acquisition function. This
  function a(x) represents the utility attributed to sampling a particular
  point x. In this context, 'utility' may mean a large reduction in
  uncertainty (exploration of new areas), or a promising large value of the
  target objective function which is to be maximized (exploitation of known
  information). An acquisition function balances this exploration-exploitation
  trade-off. At a given iteration of BO, the point x which maximizes the
  acquisition function is the one in which the target function will be
  evaluated.

  The class must have a `solver` string member, indicating the method used by
  the scipy.optimize.minimize() function.
  """
  def __init__(self, maximize_n_warmup: int, maximize_n_iter: int):
    """
    Both arguments are related to the maximization of the acquisition function.
    `maximize_n_warmup` is the number of initial warmup evaluations of the
    function, and `maximize_n_iter` is the number of different initial points
    x0 from which the maximization will start.
    """
    self.logger = logging.getLogger(__name__)
    self.n_warmup = maximize_n_warmup
    self.n_iter = maximize_n_iter

  def update_state(self, gp: GPR, history: FileDataFrame, num_iter: int) \
                   -> None:
    """Update state of the acquisition function, e.g. the Gaussian Process"""
    self.gp = gp

  @abstractmethod
  def evaluate(self, x: np.ndarray) -> float:
    """Evaluate the acquisition function in the given point"""
    pass

  def maximize(self, bounds: dict[str: tuple[float, float]]) \
               -> tuple[np.ndarray, float]:
    """
    Find the maximum of the acquisition function within the given bounds and
    with the current state. Returns both the maximizer and the maximum value.
    """
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

  def update_state(self, gp: GPR, history: FileDataFrame, num_iter: int) \
                   -> None:
    """Update state of the acquisition function, e.g. the Gaussian Process"""
    super().update_state(gp, history, num_iter)

  def evaluate(self, x: np.ndarray) -> float:
    """Evaluate the acquisition function in the given point"""
    mean, std = self.gp.predict(x, return_std=True)
    return mean + self.kappa * std


class ExpectedImprovement(AcquisitionFunction):
  solver = 'L-BFGS-B'

  def __init__(self, xi: float = 0.0, *args, **kwargs):
    self.xi = xi
    super().__init__(*args, **kwargs)

  def update_state(self, gp: GPR, history: FileDataFrame, num_iter: int) \
                   -> None:
    """Update state of the acquisition function, e.g. the Gaussian Process"""
    super().update_state(gp, history, num_iter)
    self.y_max = gp.y_train_.max()
    self.logger.debug("New EI incumbent is y = %f", self.y_max)

  def evaluate(self, x: np.ndarray) -> float:
    """Evaluate the acquisition function in the given point"""
    mean, std = self.gp.predict(x, return_std=True)

    a = (mean - self.y_max - self.xi)
    z = a / std
    return a * norm.cdf(z) + std * norm.pdf(z)


class ExpectedImprovementMachineLearning(ExpectedImprovement):
  solver = 'Nelder-Mead'

  def __init__(self, constraints: dict[str: tuple[float, float]],
               models: list[BaseEstimator], *args, **kwargs):
    super().__init__(*args, **kwargs)
    self.logger.debug("Initializing EIML with constraints=%s, models=%s, "
                      "args=%s, kwargs=%s",
                      constraints, models, args, kwargs)
    self.constraints = constraints
    self.models = dict(zip(constraints.keys(), models))

  def update_state(self, gp: GPR, history: FileDataFrame, num_iter: int) \
                   -> None:
    """Update state of the acquisition function, e.g. the Gaussian Process"""
    # Get real historical data for the training of the models
    history_df = history.get_df()
    self.logger.debug("Training ML models")
    for key in self.models:
      self.logger.debug("On column %s...", key)
      X = history_df[gp.feature_names]
      y = history_df[key]
      self.models[key].fit(X, y)
      self.logger.debug("Fitted with training data X=%s and y=%s", X.shape,
                                                                   y.shape)
    self.logger.debug("ML models trained successfully")
    super().update_state(gp, history, num_iter)

  def evaluate(self, x: np.ndarray) -> float:
    """Evaluate the acquisition function in the given point"""
    # Compute regular EI
    ret = super().evaluate(x)
    # For each constrained quantity, compute indicator of its ML prediction
    # respecting the constraints
    for key, bounds in self.constraints.items():
      lb, ub = bounds
      q_pred = self.models[key].predict(x)
      indicator = np.array([lb <= q <= ub for q in q_pred])
      ret *= indicator
    return ret
