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
import os
from typing import Dict, List, Optional, Tuple

from .dataframe import FileDataFrame


class ObjectiveFunction(ABC):
  """
  Object representing a target function to be maximized.

  A suitable target function must be able to be executed by a script (even if
  not Python) which accepts command-line parameters, and must produce an output
  file which contains (possibly among other things) the function evaluation.
  This is because this library is thought for the optimization of programs
  which must be submitted to some scheduler in order to be executed. However,
  note that nearly any function can be implemented in this form.
  The objective can also have a discrete optimization domain. If so, it must be
  created as a .csv file, and its path must be passed to the constructor as the
  `domain_file` option.
  """
  def __init__(self, domain_file: Optional[str] = None):
    self.logger = logging.getLogger(__name__)
    self.logger.debug("Initializing ObjectiveFunction with domain_file=%s",
                      domain_file)
    if domain_file is not None:
      if os.path.exists(domain_file):
        self.domain = FileDataFrame(domain_file)
      else:
        raise FileNotFoundError(f"Domain file {domain_file} not found")
    else:
      self.domain = None


  def get_approximation(self, x: np.ndarray) -> Tuple[np.ndarray, int]:
    """
    Get closest approximation of `x` from the optimization domain, wrt L2 norm

    The method fails if no domain was initialized. It returns both the
    approximation and its index in the domain
    """
    if self.domain is None:
        raise ValueError("Cannot call get_approximation() without a domain")

    min_distance = None
    approximations = []
    approximations_idxs = []

    # Recover numpy array for faster looping over rows
    df = self.domain.get_df()
    df_np = df.values
    for idx in range(df_np.shape[0]):
      row = df_np[idx, :]
      # Compute L2 distance
      dist = np.linalg.norm(x - row, 2)
      if min_distance is None or dist <= min_distance:
        if dist == min_distance:
          # One of the tied best approximations
          approximations.append(row)
          approximations_idxs.append(df.index[idx])
        else:
          # The one new best approximation
          min_distance = dist
          approximations = [row]
          approximations_idxs = [df.index[idx]]

    # If multiple, choose randomly
    ret_idx = np.random.randint(0, len(approximations_idxs))
    return approximations[ret_idx], approximations_idxs[ret_idx]


  @abstractmethod
  def execution_command(self, x: np.ndarray) -> List[str]:
    """Return the command to execute the target with the given configuration"""
    pass

  @abstractmethod
  def parse_and_evaluate(self, output_file: str) -> float:
    """Parse given output file and return the function evaluation"""
    pass

  def parse_additional_info(self, output_file: str) -> Dict[str, float]:
    """Parse given output file and return additional auxiliary information"""
    return dict()


class DummyObjective(ObjectiveFunction):
  def execution_command(self, x: np.ndarray) -> List[str]:
    """Return the command to execute the target with the given configuration"""
    return ['resources/dummy.sh', str(x[0]), str(x[1])]

  def parse_and_evaluate(self, output_file: str) -> float:
    """Parse given output file and return the function evaluation"""
    with open(output_file, 'r') as f:
      output = f.read().strip()
    return float(output)

  def parse_additional_info(self, output_file: str) -> Dict[str, float]:
    """Parse given output file and return additional auxiliary information"""
    with open(output_file, 'r') as f:
      output = f.read().strip()
    val = float(output)
    ret = {'result': val, 'result^2': val**2}
    return ret


class LigenDummyObjectiveFunction(ObjectiveFunction):
  def execution_command(self, x: np.ndarray) -> List[str]:
    """Return the command to execute the target with the given configuration"""
    return ['./ligen.sh'] + [str(_) for _ in x]

  def parse_and_evaluate(self, output_file: str) -> float:
    """Parse given output file and return the function evaluation"""
    with open(output_file, 'r') as f:
      output_list = f.read().strip().split(',')
    exe_time = float(output_list[11])
    rmsd_list = [float(_) for _ in output_list[14].split('/')]
    rmsd = np.quantile(rmsd_list, 0.75)
    objective_value = -rmsd ** 3 * exe_time
    return objective_value


class LigenReducedDummyObjective(ObjectiveFunction):
  def execution_command(self, x: np.ndarray) -> List[str]:
    """Return the command to execute the target with the given configuration"""
    return ['python', 'resources/ligen/ligen_reduced_dummy.py'] + \
           [str(_) for _ in x]

  def parse_and_evaluate(self, output_file: str) -> float:
    """Parse given output file and return the function evaluation"""
    with open(output_file, 'r') as f:
      rmsd, time = f.read().strip().split()
    val = -float(rmsd) ** 3 * float(time)
    return val

  def parse_additional_info(self, output_file: str) -> Dict[str, float]:
    """Parse given output file and return additional auxiliary information"""
    with open(output_file, 'r') as f:
      rmsd, time = f.read().strip().split()
    info = {'RMSD_0.75': rmsd}
    return info
