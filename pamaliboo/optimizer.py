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

import time

from .acquisitions import AcquisitionFunction
from .gaussian_process import DatabaseGaussianProcessRegressor as DGPR
from .jobs import JobStatus, JobSubmitter
from .objectives import ObjectiveFunction


class Optimizer:
  def __init__(self, acquisition: AcquisitionFunction,
                     bounds: dict[str: tuple[float, float]],
                     gp: DGPR,
                     job_submitter: JobSubmitter,
                     objective: ObjectiveFunction):
    self.acquisition = acquisition
    self.bounds = bounds
    self.gp = gp
    self.job_submitter = job_submitter
    self.objective = objective


  def submit_initial_points(self, n_points: int):
    pass


  def maximize(self, n_iter: int, parallelism_level: int, timeout: int):
    curr_iter = 0
    jobs_queue = {}  # TODO write to file for fault tolerance

    while curr_iter <= n_iter:
      # Find next point to be evaluated
      self.acquisition.update_state(self.gp, curr_iter)
      x_new = self.acquisition.maximize(self.bounds)

      # Submit evaluation of objective
      cmd = self.objective.execution_command(x_new)
      output_file = f"iter_{curr_iter}.stdout"
      job_id = self.job_submitter.submit(cmd, output_file)
      jobs_queue[job_id] = output_file

      # Add fake objective value to the GP
      y_fake = self.gp.predict(x_new)
      self.gp.add_point_to_database(curr_iter, x_new, y_fake)

      # Loop on finished evaluations (if any)
      for queue_id, queue_output_file in jobs_queue.items():
        if self.job_submitter.get_job_status(queue_id) == JobStatus.FINISHED:
          # TODO take output folder into account in the following lines
          y_real = self.objective.parse_and_evaluate(queue_output_file)
          # TODO remove queue_output_file
          # TODO add point to the list of real points
          self.gp.remove_point_from_database(curr_iter)
          self.gp.add_point_to_database(x_new, y_real)
          # Remove point from queue
          del jobs_queue[queue_id]

      # Fit Gaussian Process
      self.gp.fit()

      if len(jobs_queue) == parallelism_level:
        time.sleep(timeout)

      # At the end...
      curr_iter += 1
