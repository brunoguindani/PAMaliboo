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
import numpy as np
import os
import time

from .acquisitions import AcquisitionFunction
from .dataframe import FileDataFrame
from .gaussian_process import DatabaseGaussianProcessRegressor as DGPR
from .jobs import JobStatus, JobSubmitter
from .objectives import ObjectiveFunction
from .utils import join_Xy


class Optimizer:
  """
  Main class of the PAMaliboo algorithm.

  This class performs parallel asynchronous Bayesian Optimization (BO). It
  handles all other objects of the library, most of which must be provided to
  the constructor, alongside the path to an output folder where databases will
  be stored.
  The algorithm implemented is a parallel asynchronous extension of a regular
  BO algorithm. At each round, it computes the maximizer x of the acquisition
  function as normal, but instead of directly evaluating the target function
  with x, it submits the evaluation to a scheduler and moves on with the next
  iteration, after updating the Gaussian Process. A temporary fake value f' is
  produced and used by the optimization procedure, which is equal to the
  current posterior mean of the Gaussian Process evaluated in x. While the
  actual evaluation f(x) takes place, f' will be used instead. After the
  evaluation has finished, f' will be replaced by the true value f(x).
  The class keeps track of which jobs are submitted in a queue. When the size
  of the queue reaches a maximum parallelism level, a user-set timeout
  activates. This will allow for some space in the queue to be freed up before
  the next iteration.
  """
  def __init__(self, acquisition: AcquisitionFunction,
                     bounds: dict[str: tuple[float, float]],
                     gp: DGPR,
                     job_submitter: JobSubmitter,
                     objective: ObjectiveFunction,
                     output_folder: str):
    # Initialize objects
    self.acquisition = acquisition
    self.bounds = bounds
    self.gp = gp
    self.job_submitter = job_submitter
    self.objective = objective
    self.output_folder = output_folder
    self.logger = logging.getLogger(__name__)

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
    """
    Main function which performs parallel asynchronous Bayesian Optimization.

    It initializes three databases which contain useful information: the jobs
    queue, the set of real points evaluated (without fake points), and other
    optimization information such as values of the acquisition function.

    Parameters
    ----------
    n_iter: number of iterations of the algorithm
    parallelism_level: maximum number of jobs running at the same time
    timeout: waiting period in seconds if the queue is full
    """
    self.logger.debug("Initializing auxiliary dataframes in maximize()...")
    jobs_queue  = FileDataFrame(os.path.join(self.output_folder,
                                            'jobs_queue.csv'),
                                columns=['path', 'iteration'])
    real_points = FileDataFrame(os.path.join(self.output_folder,
                                             'real_points.csv'),
                                columns=self.gp.database_columns)
    other_info  = FileDataFrame(os.path.join(self.output_folder, 'info.csv'),
                                columns=['domain_idx', 'acquisition'])
    self.logger.debug("Done")

    curr_iter = 0

    while curr_iter < n_iter:
      # Check for previous interrupted runs
      db_max_idx = self.gp.database.get_df().index.max()
      if curr_iter == 0 and db_max_idx >= 0:
        curr_iter = db_max_idx + 1
        self.logger.info("Recovering iterations up to %d", curr_iter-1)
        if curr_iter >= n_iter:
          return

      self.logger.info("Starting iteration %d", curr_iter)

      # Find next point to be evaluated
      x_new, idx_appr, acq_value = self._find_next_point(curr_iter)
      # Record additional information
      other_info.add_row(curr_iter, [idx_appr, acq_value])

      # Submit evaluation of objective
      cmd = self.objective.execution_command(x_new)
      output_file = f"iter_{curr_iter}.stdout"
      job_id = self.job_submitter.submit(cmd, output_file)
      jobs_queue.add_row(job_id, [output_file, curr_iter])

      # Add fake objective value to the GP
      y_fake = self._get_fake_objective_value(x_new)
      self.logger.info("Fake prediction for %s is %f", x_new, y_fake)
      self.gp.add_point(curr_iter, x_new, y_fake)

      # Loop on finished evaluations (if any)
      queue_df = jobs_queue.get_df()
      self.logger.debug("Current queue status: %s", queue_df.to_dict())
      for queue_id, queue_row in queue_df.iterrows():
        queue_file, queue_iter = queue_row
        if self.job_submitter.get_job_status(queue_id) == JobStatus.FINISHED:
          self.logger.debug("Job %d has finished", queue_id)
          # Recover objective value from output file and the corresponding x
          output_path = os.path.join(self.job_submitter.output_folder,
                                     queue_file)
          y_real = self.objective.parse_and_evaluate(output_path)
          x_queue = self.gp.get_point(queue_iter)[:-1]
          self.logger.info("Recovered real objective value %f for job %d, "
                           "which had x=%s", y_real, queue_id, x_queue)
          # Recover additional objective information
          additional_info = self.objective.parse_additional_info(output_path)
          self.logger.debug("Recovered additional objective information: %s",
                            additional_info)
          self.acquisition.add_info(additional_info, queue_iter)
          os.remove(output_path)
          self.logger.debug("Deleted file %s", output_path)

          # Replace fake evaluation with correct one in the GP
          self.logger.debug("Updating point in GP...")
          self.gp.remove_point(queue_iter)
          self.gp.add_point(queue_iter, x_queue, y_real)

          self.logger.debug("Recording new real point...")
          new_real_point = list(join_Xy(x_queue, y_real))
          real_points.add_row(queue_iter, new_real_point)

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


  def _find_next_point(self, curr_iter: int) -> tuple[np.ndarray, int, float]:
    """
    Find next point to be evaluated. Internal use only!

    Returns the next point, its index in the optimization domain (if it exists,
    otherwise returns -curr_iter) and the maximum value of the acquisition
    function.
    """
    # Find maximizer of acquisition function
    self.acquisition.update_state(self.gp, curr_iter)
    x_new, acq_value = self.acquisition.maximize(self.bounds)
    # Round decimal places, mainly to avoid scientific notation
    x_new = np.round(x_new, 5)
    # Find discrete approximation if required
    if self.objective.domain is not None:
      x_appr, idx_appr = self.objective.get_approximation(x_new)
      self.logger.debug("Approximating %s to %s with index %d",
                        x_new, x_appr, idx_appr)
      x_new = x_appr
    else:
      idx_appr = -curr_iter

    return x_new, idx_appr, acq_value


  def _get_fake_objective_value(self, x: np.ndarray) -> float:
    """
    Return fake value of the objective function computed in x

    In this implementation, the fake value is the current posterior mean of the
    Gaussian Process evaluated in x.
    """
    return self.gp.predict(x, return_std=False)[0]
