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
import pandas as pd
import time
from typing import Dict, Tuple

from .acquisitions import AcquisitionFunction
from .dataframe import FileDataFrame
from .gaussian_process import DatabaseGaussianProcessRegressor as DGPR
from .jobs import JobStatus, JobSubmitter
from .objectives import ObjectiveFunction
from .utils import dict_to_array, join_Xy, numpy_to_str, str_to_numpy


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
  activates, to allow for some jobs in the queue to finish. The timeout will
  repeat, and no new jobs will be submitted, until some space in the queue is
  freed up.
  """
  history_filename = 'history.csv'
  other_info_filename = 'info.csv'
  queue_filename = 'queue.csv'
  output_file_fmt = 'iter_{}.stdout'

  def __init__(self, acquisition: AcquisitionFunction,
                     bounds: Dict[str, Tuple[float, float]],
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


  def initialize(self, init_history_path: str) -> None:
    """Initialize optimizer and GP with history from given file"""
    self.logger.info("Initializing optimizer...")
    if len(self.gp.database) > 0:
      self.logger.info("GP database already contains points: input database "
                       "ignored")
    else:
      # Initialize history database from given file
      self.logger.info("Setting initial points...")
      df = pd.read_csv(init_history_path, index_col=FileDataFrame.index_name)
      _ = FileDataFrame(os.path.join(self.output_folder,
                                     self.history_filename), data=df)
      df_gp = df[self.gp.database_columns]
      df_gp.to_csv(self.gp.database_path,
                   index_label=FileDataFrame.index_name)
    self.logger.debug("Fitting GP...")
    self.gp.fit()
    self.logger.debug("Done")
    self.logger.info("Initialization complete")


  def maximize(self, n_iter: int, parallelism_level: int, timeout: float) \
               -> None:
    """
    Main function which performs parallel asynchronous Bayesian Optimization.

    It initializes three databases which are stored in the output folder: the
    jobs queue, the history of real points evaluated alongside the additional
    job output information (if any), and other optimization information such as
    values of the acquisition function.

    Parameters
    ----------
    `n_iter`: number of iterations of the algorithm
    `parallelism_level`: maximum number of jobs running at the same time
    `timeout`: waiting period in seconds if the queue is full
    """
    self.logger.debug("Initializing auxiliary dataframes in maximize()...")
    self.history = FileDataFrame(os.path.join(self.output_folder,
                                              self.history_filename))
    jobs_queue = FileDataFrame(os.path.join(self.output_folder,
                                            self.queue_filename),
                               columns=['path', 'iteration', 'x'])
    other_info = FileDataFrame(os.path.join(self.output_folder,
                                            self.other_info_filename),
                               columns=['domain_idx', 'acquisition',
                                        'train_MAPE', 'optimizer_time',
                                        'processing_time'])
    self.logger.debug("Done")

    curr_iter = 0
    start_time = time.time()

    while curr_iter < n_iter or len(jobs_queue) > 0:
      curr_iter_start_time = time.time()
      # Check for previous interrupted runs
      db_max_idx = self.gp.database.get_df().index.max()
      if curr_iter == 0 and db_max_idx >= 0:
        curr_iter = db_max_idx + 1
        self.logger.info("Recovering iterations up to %d", curr_iter-1)
        if curr_iter >= n_iter:
          return

      self.logger.debug("Recovering finished jobs (if any)")
      queue_df = jobs_queue.get_df()
      self.logger.debug("Current queue status: %s", queue_df.to_dict())
      recovered_queue_ids = []
      for queue_id, queue_row in queue_df.iterrows():
        queue_file, queue_iter, x_queue = queue_row
        x_queue = str_to_numpy(x_queue)
        if self.job_submitter.get_job_status(queue_id) == JobStatus.FINISHED:
          self.logger.debug("Job %d has finished", queue_id)

          # Recover objective value from output file and the corresponding x
          self._recover_and_insert_value(queue_file, queue_iter, x_queue)

          self.logger.debug("Removing job %d from queue...", queue_id)
          jobs_queue.remove_row(queue_id)
          recovered_queue_ids.append(queue_id)

      self.logger.debug("Recovering of finished jobs completed")

      # If a new true point was recorded, the next point will be chosen only
      # according to the true knowledge: we therefore remove fake evaluations
      # from the GP. Note that such points will still be present in the queue
      if recovered_queue_ids:
        self.logger.debug("Real objective value(s) have been recovered: "
                          "deleting all fake values...")
        for queue_id, queue_row in queue_df.iterrows():
          queue_iter = queue_row['iteration']
          # check that the point was not one whose true value we just recovered
          if (queue_id not in recovered_queue_ids and
              queue_iter in self.gp.database):
            self.gp.remove_point(queue_iter)
        self.logger.debug("Fake points deleted")

      # Skip iteration if queue is full
      if curr_iter < n_iter and len(jobs_queue) == parallelism_level:
        self.logger.debug("Maximum parallelism level reached: sleeping for "
                          "%.2f seconds...", timeout)
        time.sleep(timeout)
        continue
      # Skip iteration if the last iteration was reached
      elif curr_iter == n_iter:
        if len(jobs_queue) > 0:
          self.logger.debug("Unfinished jobs at end of algorithm: sleeping "
                            "for %.2f seconds...", timeout)
          time.sleep(timeout)
        continue

      self.logger.info("Starting iteration %d", curr_iter)

      # Find next point to be evaluated
      x_new, idx_appr, acq_value = self._find_next_point(curr_iter)
      # Record additional information
      error = getattr(self.acquisition, 'train_MAPE', None)  # meh...
      curr_time = time.time() - start_time
      processing_time = time.time() - curr_iter_start_time
      other_info.add_row(curr_iter, [idx_appr, acq_value, error, curr_time,
                                     processing_time])

      # Submit evaluation of objective
      cmd = self.objective.execution_command(x_new)
      output_file = self.output_file_fmt.format(curr_iter)
      job_id = self.job_submitter.submit(cmd, output_file)
      self.logger.debug("Job submitted with id %d", job_id)
      jobs_queue.add_row(job_id, [output_file, curr_iter, numpy_to_str(x_new)])

      # Add fake objective value to the GP
      y_fake = self._get_fake_objective_value(x_new)
      self.logger.info("Fake prediction for %s is %f", x_new, y_fake)
      self.gp.add_point(curr_iter, x_new, y_fake)

      # Fit Gaussian Process
      self.logger.debug("Updating GP...")
      self.gp.fit()

      # At the end...
      self.logger.debug("End of iteration %d", curr_iter)
      curr_iter += 1

    # Clean up after ending the loop
    self.history = None
    self.logger.info("End of optimization algorithm")


  def _recover_and_insert_value(self, queue_file: str, queue_iter: int,
                                x_queue: np.ndarray) -> None:
    """
    Recover objective value and additional information. Internal use only!

    Parses `queue_file` to get the value of the objective function and any
    additional information (deleting the file in the process). It then updates
    class databases with the recovered values alongside the corresponding
    configuration value `x_queue`. The new rows will have index `queue_iter` in
    the databases.
    """
    
    output_path = os.path.join(self.job_submitter.output_folder, queue_file)
    y_real = self.objective.parse_and_evaluate(output_path)
    self.logger.info("Recovered real objective value %f, which had x=%s",
                     y_real, x_queue)
    # Recover additional objective information
    additional_info = self.objective.parse_additional_info(output_path)
    self.logger.debug("Recovered additional objective information: %s",
                      additional_info)
    os.remove(output_path)
    self.logger.debug("Deleted file %s", output_path)

    # Replace fake evaluation with correct one in the GP
    self.logger.debug("Updating point in GP...")
    if queue_iter in self.gp.database:
      self.gp.remove_point(queue_iter)
    self.gp.add_point(queue_iter, x_queue, y_real)

    self.logger.debug("Recording new real point...")
    new_real_point = join_Xy(x_queue, y_real)
    new_add_info = dict_to_array(additional_info)
    new_row = np.hstack((new_real_point, new_add_info))
    self.history.add_row(queue_iter, new_row)


  def _find_next_point(self, curr_iter: int) -> Tuple[np.ndarray, int, float]:
    """
    Find next point to be evaluated. Internal use only!

    Returns the next point, its index in the optimization domain (if it exists,
    otherwise returns -`curr_iter`) and the maximum value of the acquisition
    function.
    """
    # Find maximizer of acquisition function
    self.logger.debug("Updating state of acquisition function...")
    self.acquisition.update_state(self.gp, self.history, curr_iter)
    self.logger.debug("Done")
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
    Return fake value of objective function computed in `x`. Internal use only!

    In this implementation, the fake value is the current posterior mean of the
    Gaussian Process evaluated in `x`.
    """
    return self.gp.predict(x, return_std=False)[0]



class OptimizerSimulator(Optimizer):
  """
  Class that simulates evaluations of the objective function.

  This class is similar to `Optimizer`, but is meant for use with
  `SimulatorSubmitter` and objective functions whose execution time is known a
  priori, e.g., from a dataset table. Rather than measuring and recording the
  actual wallclock time elapsed for optimization, it does so with a clock
  (`the_clock`) which is set to 0 at the beginning of the optimization
  procedure and is advanced by the correct amount of time when necessary. This
  allows a large speed-up when simulating optimization experiments with a time-
  consuming objective function.
  This kind of simulation also requires to keep a `countdown` for each pending
  job, which are recorded as a column of `jobs_queue`.
  Note that unlike `Optimizer`, this class cannot be fault-tolerant.
  """
  def maximize(self, n_iter: int, parallelism_level: int, timeout: float) \
               -> None:
    """
    Main function which performs parallel asynchronous Bayesian Optimization.

    It initializes three databases which are stored in the output folder: the
    jobs queue, the history of real points evaluated alongside the additional
    job output information (if any), and other optimization information such as
    values of the acquisition function.

    Parameters
    ----------
    `n_iter`: number of iterations of the algorithm
    `parallelism_level`: maximum number of jobs running at the same time
    `timeout`: unused
    """
    self.logger.debug("Initializing auxiliary dataframes in maximize()...")
    self.history = FileDataFrame(os.path.join(self.output_folder,
                                              self.history_filename))
    jobs_queue = FileDataFrame(os.path.join(self.output_folder,
                                            self.queue_filename),
                               columns=['path', 'iteration', 'x', 'countdown'])
    other_info = FileDataFrame(os.path.join(self.output_folder,
                                            self.other_info_filename),
                               columns=['domain_idx', 'acquisition',
                                        'train_MAPE', 'optimizer_time',
                                        'processing_time'])
    self.logger.debug("Done")

    curr_iter = 0
    the_clock = 0

    while curr_iter < n_iter or len(jobs_queue) > 0:
      # Record time at start of iteration
      self.logger.debug("Starting new loop iteration")
      self.logger.debug("The Clock = %f", the_clock)
      curr_iter_start_time = time.time()

      self.logger.debug("Recovering finished jobs (if any)")
      queue_df = jobs_queue.get_df()
      self.logger.debug("Current queue status:\n%s", queue_df)
      recovered_queue_ids = []
      for queue_id, queue_row in queue_df.iterrows():
        queue_file, queue_iter, x_queue, queue_countdown = queue_row
        x_queue = str_to_numpy(x_queue)
        if queue_countdown == 0:
          self.logger.debug("Job %d has finished", queue_id)

          # Recover objective value from output file and the corresponding x
          self._recover_and_insert_value(queue_file, queue_iter, x_queue)

          self.logger.debug("Removing job %d from queue...", queue_id)
          jobs_queue.remove_row(queue_id)
          recovered_queue_ids.append(queue_id)

      self.logger.debug("Recovering of finished jobs completed")

      # Remove fake evaluations from the GP (similar to `Optimizer`)
      if recovered_queue_ids:
        self.logger.debug("Real objective value(s) have been recovered: "
                          "deleting all fake values...")
        for queue_id, queue_row in queue_df.iterrows():
          queue_iter = queue_row['iteration']
          # check that the point was not one whose true value we just recovered
          if (queue_id not in recovered_queue_ids and
              queue_iter in self.gp.database):
            self.gp.remove_point(queue_iter)
        self.logger.debug("Fake points deleted")

      if curr_iter < n_iter and len(jobs_queue) == parallelism_level:
        # Skip iteration if queue is full
        delta_queue = jobs_queue.get_df()['countdown'].min()
        self.logger.debug("Maximum parallelism level reached; next "
                          "evaluation will finish in %f seconds", delta_queue)
      elif curr_iter == n_iter:
        # Skip iteration if the last iteration was reached
        delta_queue = jobs_queue.get_df()['countdown'].min()
        self.logger.debug("Unfinished jobs at end of algorithm; next "
                          "evaluation will finish in %f seconds", delta_queue)
      else:
        # Peform actual iteration
        self.logger.info("Starting algorithm iteration %d", curr_iter)

        # Find next point to be evaluated
        x_new, idx_appr, acq_value = self._find_next_point(curr_iter)
        # Record additional information
        error = getattr(self.acquisition, 'train_MAPE', None)  # meh...
        processing_time = time.time() - curr_iter_start_time
        curr_time = the_clock + processing_time
        self.logger.debug("Recording clock time = %f", curr_time)
        other_info.add_row(curr_iter, [idx_appr, acq_value, error, curr_time,
                                       processing_time])

        # Submit evaluation of objective
        cmd = self.objective.execution_command(x_new)
        output_file = self.output_file_fmt.format(curr_iter)
        job_id = self.job_submitter.submit(cmd, output_file)
        self.logger.debug("Job submitted with id %d", job_id)
        output_path = os.path.join(self.job_submitter.output_folder,
                                   output_file)
        new_add_info = self.objective.parse_additional_info(output_path)
        new_exec_time = new_add_info['evaluation_time']
        self.logger.debug("Evaluation time will be %f seconds", new_exec_time)
        jobs_queue.add_row(job_id, [output_file, curr_iter,
                                    numpy_to_str(x_new), new_exec_time])

        # Add fake objective value to the GP
        y_fake = self._get_fake_objective_value(x_new)
        self.logger.info("Fake prediction for %s is %f", x_new, y_fake)
        self.gp.add_point(curr_iter, x_new, y_fake)

        # Fit Gaussian Process
        self.logger.debug("Updating GP...")
        self.gp.fit()

        # At the end of a proper (configuration-choosing) iteration...
        self.logger.debug("End of iteration %d", curr_iter)
        curr_iter += 1
        delta_queue = 0

      # At the end of any kind of iteration...
      ## Compute elapsed time
      delta_algo = time.time() - curr_iter_start_time
      delta = max(delta_queue, delta_algo)
      self.logger.debug("Advancing The Clock and queue countdowns by "
                        "%f seconds", delta)
      the_clock += delta
      ## Advance countdowns for individual jobs by `delta`
      queue_df = jobs_queue.get_df()
      queue_df['countdown'] = (queue_df['countdown'] - delta).clip(lower=0)
      jobs_queue.set_df(queue_df)
      self.logger.debug("End of loop iteration")


    # Clean up after ending the loop
    self.history = None
    self.logger.info("End of optimization algorithm")
