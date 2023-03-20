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
import numpy as np
from scipy.stats import norm
from sklearn.gaussian_process import GaussianProcessRegressor as GPR


class AcquisitionFunction(ABC):
  def update_state(self, gp: GPR, num_iter: int) -> None:
    self.gp = gp

  @abstractmethod
  def evaluate(self, x: np.ndarray) -> float:
    pass


class UpperConfidenceBound(AcquisitionFunction):
  def __init__(self, kappa: float = 2.576):
    self.kappa = kappa

  def update_state(self, gp: GPR, num_iter: int) -> None:
    super().update_state(gp, num_iter)

  def evaluate(self, x: np.ndarray) -> float:
    mean, std = self.gp.predict(x, return_std=True)
    return mean + self.kappa * std


class ExpectedImprovement(AcquisitionFunction):
  def __init__(self, xi: float = 0.0):
    self.xi = xi

  def update_state(self, gp: GPR, num_iter: int) -> None:
    super().update_state(gp, num_iter)
    self.y_max = gp.y_train_.max()

  def evaluate(self, x: np.ndarray) -> float:
    mean, std = self.gp.predict(x, return_std=True)

    a = (mean - self.y_max - self.xi)
    z = a / std
    return a * norm.cdf(z) + std * norm.pdf(z)
