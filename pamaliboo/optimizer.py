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

import os
import time

from .acquisitions import AcquisitionFunction
from .dataframe import FileDataFrame
from .gaussian_process import DatabaseGaussianProcessRegressor as DGPR
from .jobs import JobStatus, JobSubmitter
from .objectives import ObjectiveFunction
from .utils import join_Xy


class Optimizer:
  def __init__(self, acquisition: AcquisitionFunction,
                     bounds: dict[str: tuple[float, float]],
                     gp: DGPR,
                     job_submitter: JobSubmitter,
                     objective: ObjectiveFunction,
                     output_folder: str):
    self.acquisition = acquisition
    self.bounds = bounds
    self.gp = gp
    self.job_submitter = job_submitter
    self.objective = objective
    self.output_folder = output_folder
    os.makedirs(self.output_folder, exist_ok=True)


  def submit_initial_points(self, n_points: int, timeout: int):
    # TODO
    pass


  def maximize(self, n_iter: int, parallelism_level: int, timeout: int):
    # Initialize dataframes
    jobs_queue = FileDataFrame(os.path.join(self.output_folder,
                                            'jobs_queue.csv'),
                               columns=['path', 'iteration'])
    real_points = FileDataFrame(os.path.join(self.output_folder,
                                             'real_points.csv'),
                                columns=self.gp.database_columns)

    curr_iter = 0

    while curr_iter <= n_iter:
      # Find next point to be evaluated
      self.acquisition.update_state(self.gp, curr_iter)
      # TODO collect acq_value in a database (?)
      x_new, acq_value = self.acquisition.maximize(self.bounds)

      # Submit evaluation of objective
      cmd = self.objective.execution_command(x_new)
      output_file = f"iter_{curr_iter}.stdout"
      job_id = self.job_submitter.submit(cmd, output_file)
      jobs_queue.add_row(job_id, [output_file, curr_iter])

      # Add fake objective value to the GP
      y_fake = self.gp.predict(x_new, return_std=False)[0]
      self.gp.add_point(curr_iter, x_new, y_fake)

      # Loop on finished evaluations (if any)
      queue_df = jobs_queue.get_df()
      for queue_id, queue_row in queue_df.iterrows():
        queue_file, queue_iter = queue_row
        if self.job_submitter.get_job_status(queue_id) == JobStatus.FINISHED:
          # Get objective value from output file, then remove the file
          output_path = os.path.join(self.job_submitter.output_folder,
                                     queue_file)
          y_real = self.objective.parse_and_evaluate(output_path)
          os.remove(output_path)
          # Replace fake evaluation with correct one in the GP
          self.gp.remove_point(queue_iter)
          self.gp.add_point(queue_iter, x_new, y_real)
          # Record real point in the corresponding dataframe
          new_real_point = list(join_Xy(x_new, y_real))
          real_points.add_row(queue_iter, new_real_point)
          # Remove point from queue
          jobs_queue.remove_row(queue_id)

      # Fit Gaussian Process
      self.gp.fit()

      if len(jobs_queue) == parallelism_level:
        time.sleep(timeout)

      # At the end...
      curr_iter += 1
