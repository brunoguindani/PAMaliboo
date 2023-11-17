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
from multiprocessing import Pool
import numpy as np
import os
import pandas as pd
from sklearn.gaussian_process.kernels import Matern
from sklearn.linear_model import Ridge
import time

from pamaliboo.acquisitions import ExpectedImprovementMachineLearning as EIML
from pamaliboo.batch import BatchExecutor
from pamaliboo.gaussian_process import DatabaseGaussianProcessRegressor as DGPR
from pamaliboo.jobs import HyperqueueJobSubmitter
from pamaliboo.objectives import LigenSynthDummyObjective
from pamaliboo.optimizer import Optimizer


# Campaign parameters
parallelism = 10
num_runs = 10
num_iter = 1000
n_init = 20
root_rng_seed = 20230524
root_output_folder = f'outputs_ligen_synth_init{n_init}'
ml_models = [Ridge()]

# Other parameters
opt_bounds = {'ALIGN_SPLIT': [8, 72.01], 'OPTIMIZE_SPLIT': [8, 72.01],
              'OPTIMIZE_REPS': [1, 5.01], 'CUDA_THREADS': [32, 256.01],
              'N_RESTART': [256, 1024.01], 'CLIPPING': [10, 256.01],
              'SIM_THRESH': [1, 4.01], 'BUFFER_SIZE': [1048576, 52428800.01]}
opt_constraints = {'RMSD_0.75': (0, 2.1)}
features = list(opt_bounds.keys())
domain = os.path.join('resources', 'ligen', 'ligen_synth_domain.csv')
timeout = 1

# Initialize and set relevant stuff
domain_df = pd.read_csv(domain, index_col='index')
logging.basicConfig(level=logging.DEBUG)


# Function for a single experiment
def run_experiment(rng):
    print(f"New run with parallelism {par} and RNG seed {rng}")
    # Create output folder for this experiment
    output_folder = os.path.join(root_output_folder, f'par_{par}',
                                                     f'rng_{rng}')
    os.makedirs(output_folder, exist_ok=True)

    # Initialize library objects
    acq = EIML(constraints=opt_constraints, models=ml_models,
               train_periodicity=3, pickle_folder=None,
               maximize_n_warmup=10, maximize_n_iter=100)
    kernel = Matern(nu=2.5)
    obj = LigenSynthDummyObjective(domain_file=domain)
    job_submitter = HyperqueueJobSubmitter(output_folder)
    batch_ex = BatchExecutor(job_submitter, obj)
    gp_path = os.path.join(output_folder, 'gp_database.csv')
    gp = DGPR(gp_path, feature_names=features, kernel=kernel, normalize_y=True)
    optimizer = Optimizer(acq, opt_bounds, gp, job_submitter, obj,
                          output_folder)

    # Get random initial points
    np.random.seed(rng)
    df_init = domain_df.sample(n_init)

    # Run initial points
    res = batch_ex.execute(df_init, timeout=timeout)
    new_idx = pd.Index([-1]*res.shape[0])
    res.set_index(new_idx, inplace=True)
    init_history = os.path.join(output_folder, 'init.csv')
    res.to_csv(init_history, index_label='index')

    # Perform optimization
    optimizer.initialize(init_history)
    optimizer.maximize(n_iter=num_iter, parallelism_level=par, timeout=timeout)
    print("Run completed", flush=True)


print("Current time:", time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
      flush=True)

# Loop over paralellism levels and RNG seeds
for par in [parallelism, 1]:
  # Initialize RNG seeds
  rng_seeds = [root_rng_seed+i for i in range(num_runs)]
  # Further seeds for independent sequential runs
  if par == 1 and parallelism > 1:
    for r in list(rng_seeds):
      group_seeds = [r] + [10*r+i for i in range(parallelism-1)]
      # Run such experiments in parallel
      with Pool(parallelism) as pool:
        pool.map(run_experiment, group_seeds)
      print("Parallel batch completed\n" + 40*"-" + "\n\n")
  else:
    for rng in rng_seeds:
      run_experiment(rng)

print("Current time:", time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
      flush=True)
