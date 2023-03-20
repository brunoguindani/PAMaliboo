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

from .acquisitions import AcquisitionFunction
from .gaussian_process import DatabaseGaussianProcessRegressor as DGPR
from .jobs import JobSubmitter
from .objectives import ObjectiveFunction


class Optimizer:
  def __init__(self, acquisition: AcquisitionFunction,
                     bounds: dict[str: tuple[float, float]],
                     gp: DGPR,
                     job_summitter: JobSubmitter,
                     objective: ObjectiveFunction):
    self.acquisition = acquisition
    self.bounds = bounds
    self.gp = gp
    self.job_summitter = job_summitter
    self.objective = objective


  def submit_initial_points(self, n_points: int):
    pass


  def maximize(self, n_iter: int, parallelism_level: int):
    curr_iter = 0
    jobs_queue = []  # TODO write to file for fault tolerance

    while curr_iter <= n_iter:
      # Find next point to be evaluated
      self.acquisition.update_state(self.gp, curr_iter)
      x_new = self.acquisition.maximize(self.bounds)

      # Submit evaluation of objective
      cmd = self.objective.execution_command(x_new)
      job_id = self.job_summitter.submit(cmd)
      jobs_queue.append(job_id)

      # Add fake objective value to the GP
      y_fake = self.gp.predict(x_new)
      self.gp.add_point_to_database(curr_iter, x_new, y_fake)

      # for each finished evaluation (if any)...
      # TODO


      # At the end...
      curr_iter += 1
