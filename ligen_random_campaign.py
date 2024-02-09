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

from pamaliboo.jobs import HyperqueueJobSubmitter, JobStatus
from pamaliboo.objectives import LigenSynthDummyObjective


# Campaign parameters
parallelism = 10
num_runs = 10
n_init = 20
num_iter = 1000 + n_init
root_rng_seed = 20230524
timeout = 15
root_output_folder = f'outputs_ligen_synth_init{n_init}'
domain = os.path.join('resources', 'ligen', 'ligen_synth_domain.csv')

# Initialize and set other relevant stuff
rng_seeds = [root_rng_seed+i for i in range(num_runs)]
domain_df = pd.read_csv(domain, index_col='index')
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG)
objective = LigenSynthDummyObjective()


for rng in rng_seeds:
  print(f"New run with RNG seed {rng}")
  start_time = time.time()
  # Folder and file names for this experiment
  output_folder = os.path.join(root_output_folder, 'par_random', f'rng_{rng}')
  history_file = os.path.join(output_folder, 'history.csv')
  info_file = os.path.join(output_folder, 'info.csv')
  os.makedirs(output_folder, exist_ok=True)
  # Initialize relevant objects
  job_submitter = HyperqueueJobSubmitter(output_folder)
  df_queue = pd.DataFrame(columns=['idx', 'file'])
  df_history = pd.DataFrame()
  df_info = pd.DataFrame(columns=['optimizer_time'])

  # Get all random configurations
  np.random.seed(rng)
  all_configs = domain_df.sample(num_iter)

  curr_iter = 0
  while curr_iter < num_iter or len(df_queue) > 0:
    # Loop on jobs to find finished ones
    for job_id, [config_idx, job_file] in df_queue.iterrows():
      if job_submitter.get_job_status(job_id) == JobStatus.FINISHED:
        logger.debug("Job %d has finished", job_id)
        # Extract information from job file
        output_path = os.path.join(output_folder, job_file)
        info = {'target': objective.parse_and_evaluate(output_path)}
        add_info = objective.parse_additional_info(output_path)
        info.update(add_info)
        logger.debug("Recovered information: %s", info)
        # Update DataFrames and clean up files
        df_queue.drop(job_id, inplace=True)
        df_history.loc[config_idx, info.keys()] = info.values()
        df_info.loc[config_idx, 'optimizer_time'] = time.time() - start_time
        logger.debug("Iteration: %d, time elapsed: %d",
                     curr_iter, df_info.loc[config_idx, 'optimizer_time'])
        os.remove(output_path)

    # Skip iteration if queue is full
    if curr_iter < num_iter and len(df_queue) == parallelism:
      logger.debug("Maximum parallelism level reached: sleeping for "
                   "%.2f seconds...", timeout)
      time.sleep(timeout)
      continue
    # Skip iteration if the last iteration was reached
    elif curr_iter == num_iter:
      if len(df_queue) > 0:
        logger.debug("Unfinished jobs at end of algorithm: sleeping "
                     "for %.2f seconds...", timeout)
        time.sleep(timeout)
      continue

    conf = all_configs.iloc[curr_iter]
    cmd = objective.execution_command(conf)
    stdout_file = f'batch_{curr_iter}.stdout'
    job_id = job_submitter.submit(cmd, stdout_file)
    df_queue.loc[job_id] = [curr_iter, stdout_file]

    curr_iter += 1

    if not df_queue.empty:
      df_history.to_csv(history_file, index_label='index')
      df_info.to_csv(info_file, index_label='index')

  # Save data one last time
  df_history.to_csv(history_file, index_label='index')
  df_info.to_csv(info_file, index_label='index')
  print("Run completed")
  print("Current time:", time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
        flush=True)

print("Campaign completed; bye!")
