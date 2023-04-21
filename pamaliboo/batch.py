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
import pandas as pd
import time

from .jobs import JobStatus, JobSubmitter
from .objectives import ObjectiveFunction


class BatchExecutor:
  """
  Utility class for batch execution.

  It is initialized with a `JobSubmitter` and an `ObjectiveFunction`. It allows
  submitting multiple jobs at once to the job scheduler, by passing a
  `pandas.DataFrame` containing the job configurations to be submitted.
  Results are then returned in `DataFrame` form. For instance, this class is
  useful when creating the initial points database required by the `Optimizer`
  class, by saving the returned `DataFrame` to file.
  """
  def __init__(self, job_submitter: JobSubmitter,
                     objective: ObjectiveFunction):
    self.logger = logging.getLogger(__name__)
    self.job_submitter = job_submitter
    self.objective = objective

    self.output_folder = self.job_submitter.output_folder
    self.logger.debug("Recovered output folder %s from job submitter",
                      self.output_folder)


  def execute(self, config_df: pd.DataFrame, timeout: float) -> pd.DataFrame:
    """
    Perform batch execution of any number of given configurations

    Each row in `config_df` contains a configuration to be submitted. Then,
    once every `timeout` seconds, the status of the submitted jobs will be
    checked. When all jobs are finished, the results are collected and returned
    in a `pandas.DataFrame`. It contains the objective values and additional
    information, if any, parsed from the output file.
    """
    self.logger.info("Performing batch execution...")
    jobs_queue = pd.DataFrame(columns=['idx', 'file'])

    # Loop over configurations
    for idx, conf in config_df.iterrows():
      # Build execution command and submit job
      cmd = self.objective.execution_command(conf)
      output_file = f'batch_{idx}.stdout'
      job_id = self.job_submitter.submit(cmd, output_file)
      # Add information to jobs queue
      jobs_queue.loc[job_id] = [idx, output_file]

    # Wait until all jobs are finished
    while not self.all_finished(jobs_queue.index):
      self.logger.debug("Unfinished jobs: sleeping for %.2f seconds...",
                        timeout)
      time.sleep(timeout)

    self.logger.info("All jobs have finished: collecting results...")
    output_df = config_df.copy()

    # Loop over jobs
    for jid, [jidx, jfile] in jobs_queue.iterrows():
      # Recover objective value and additional information from output file
      output_path = os.path.join(self.job_submitter.output_folder, jfile)
      info = {'target': self.objective.parse_and_evaluate(output_path)}
      add_info = self.objective.parse_additional_info(output_path)
      info.update(add_info)
      self.logger.debug("Recovered information from job %d: %s", jid, info)

      # Write to output dataframe
      output_df.loc[jidx, info.keys()] = info.values()

      os.remove(output_path)
      self.logger.debug("Deleted file %s", output_path)

    self.logger.info("Collected result matrix with shape %s", output_df.shape)
    return output_df


  def all_finished(self, jobs_ids: pd.Index) -> bool:
    """Check whether or not ALL jobs in `jobs_ids` are finished"""
    for jid in jobs_ids:
      status = self.job_submitter.get_job_status(jid)
      if status in (JobStatus.CANCELED, JobStatus.FAILED):
        raise RuntimeError(f"Job {jid} in batch execution has status {status}")
      if status != JobStatus.FINISHED:
        return False
    return True
