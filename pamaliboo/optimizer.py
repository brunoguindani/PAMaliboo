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

import logging
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
    # Initialize logger
    self.logger = logging.getLogger(__name__)

    # Set other objects
    self.acquisition = acquisition
    self.bounds = bounds
    self.gp = gp
    self.job_submitter = job_submitter
    self.objective = objective
    self.output_folder = output_folder

    # Create output folder
    if os.path.exists(self.output_folder):
      self.logger.debug("Output folder %s already exists", self.output_folder)
    else:
      os.makedirs(self.output_folder)
      self.logger.debug("Created output folder %s", self.output_folder)


  def submit_initial_points(self, n_points: int, timeout: int):
    # TODO
    pass


  def maximize(self, n_iter: int, parallelism_level: int, timeout: int):
    self.logger.debug("Initializing auxiliary dataframes in maximize()")
    jobs_queue = FileDataFrame(os.path.join(self.output_folder,
                                            'jobs_queue.csv'),
                               columns=['path', 'iteration'])
    real_points = FileDataFrame(os.path.join(self.output_folder,
                                             'real_points.csv'),
                                columns=self.gp.database_columns)
    self.logger.debug("Done")

    curr_iter = 0

    while curr_iter <= n_iter:
      self.logger.info("Starting iteration %d", curr_iter)
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
      self.logger.info("Fake prediction for %s is %f", x_new, y_fake)
      self.gp.add_point(curr_iter, x_new, y_fake)

      # Loop on finished evaluations (if any)
      queue_df = jobs_queue.get_df()
      self.logger.debug("Current queue status: %s", queue_df.to_dict())
      for queue_id, queue_row in queue_df.iterrows():
        queue_file, queue_iter = queue_row
        if self.job_submitter.get_job_status(queue_id) == JobStatus.FINISHED:
          self.logger.debug("Job %d has finished", queue_id)
          # Get objective value from output file, then remove the file
          output_path = os.path.join(self.job_submitter.output_folder,
                                     queue_file)
          y_real = self.objective.parse_and_evaluate(output_path)
          self.logger.info("Recovered real objective value %f for job %d",
                           y_real, queue_id)
          os.remove(output_path)
          self.logger.debug("Deleted file %s", output_path)
          # Replace fake evaluation with correct one in the GP
          self.logger.debug("Updating point in GP...")
          self.gp.remove_point(queue_iter)
          self.gp.add_point(queue_iter, x_new, y_real)
          # Record real point in the corresponding dataframe
          self.logger.debug("Recording new real point...")
          new_real_point = list(join_Xy(x_new, y_real))
          real_points.add_row(queue_iter, new_real_point)
          # Remove point from queue
          self.logger.debug("Removing job %d from queue...", queue_id)
          jobs_queue.remove_row(queue_id)

      # Fit Gaussian Process
      self.logger.debug("Updating GP...")
      self.gp.fit()

      if len(jobs_queue) == parallelism_level:
        self.logger.debug("Maximum parallelism level reached: sleeping for %d "
                          "seconds", timeout)
        time.sleep(timeout)

      # At the end...
      self.logger.debug("End of iteration %d", curr_iter)
      curr_iter += 1
