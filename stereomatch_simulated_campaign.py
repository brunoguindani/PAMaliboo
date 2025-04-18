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

from pamaliboo.acquisitions import ExpectedImprovementMachineLearning as EIML
from pamaliboo.batch import BatchExecutor
from pamaliboo.gaussian_process import DatabaseGaussianProcessRegressor as DGPR
from pamaliboo.jobs import StereomatchSimulatorSubmitter
from pamaliboo.objectives import StereomatchSimulatedObjective
from pamaliboo.optimizer import OptimizerSimulator


# Campaign parameters
parallelism = 4
num_runs = 10
num_iter_seq = 30
n_init = 3
root_rng_seed = 20230524  # int(sys.argv[1])
pool_seq_parallelism = 4
Tmax = 200  # 110, 125, 140, 200
root_output_folder = os.path.join('outputs',
                                 f'stereomatch10_T{Tmax}_p{parallelism}_init{n_init}')
os.makedirs(root_output_folder, exist_ok=True)
log_file = os.path.basename(root_output_folder) + '.log'
log_file_path = os.path.join(root_output_folder, log_file)
ml_models = [Ridge()]
all_parallelism_levels = [parallelism, 1]

# Other parameters
opt_bounds = {'confidence': [14, 64.01], 'hypo_step': [1, 3.01],
              'max_arm_length': [1, 16.01], 'num_threads': [1, 32.01]}
opt_constraints = {'time': (0, Tmax)}
features = list(opt_bounds.keys())
domain = os.path.join('resources', 'stereomatch_domain.csv')
table = os.path.join('resources', 'stereomatch_10_table.csv')
timeout = 1

# Initialize and set relevant stuff
domain_df = pd.read_csv(domain, index_col='index')
logging.basicConfig(format='%(asctime)s|%(name)s|%(levelname)s|%(message)s',
                    level=logging.DEBUG, datefmt='%Y-%m-%dT%H:%M:%S',
                    handlers=[logging.FileHandler(log_file_path),
                              logging.StreamHandler()])
logger = logging.getLogger(__name__)


# Function for a single experiment
def run_experiment(rng):
    logger.info("New run with RNG seed %d", rng)
    # Create output folder for this experiment
    output_folder = os.path.join(root_output_folder, f'par_{par}',
                                                     f'rng_{rng}')
    os.makedirs(output_folder, exist_ok=True)

    # Initialize library objects
    acq = EIML(# error_init=errinit, error_maxiter=trans,
               constraints=opt_constraints, models=ml_models,
               train_periodicity=3, pickle_folder=None,
               maximize_n_warmup=10, maximize_n_iter=100)
    kernel = Matern(nu=2.5)
    obj = StereomatchSimulatedObjective(domain_file=domain)
    job_submitter = StereomatchSimulatorSubmitter(output_folder, table)
    batch_ex = BatchExecutor(job_submitter, obj)
    gp_path = os.path.join(output_folder, 'gp_database.csv')
    gp = DGPR(gp_path, feature_names=features, kernel=kernel, normalize_y=True)
    optimizer = OptimizerSimulator(acq, opt_bounds, gp, job_submitter, obj,
                                   output_folder)

    # Get random initial points
    np.random.seed(rng)
    df_init = domain_df.sample(n_init*par)

    # Run initial points
    res = batch_ex.execute(df_init, timeout=timeout)
    new_idx = pd.Index([-1]*res.shape[0])
    res.set_index(new_idx, inplace=True)
    init_history = os.path.join(output_folder, 'init.csv')
    res.to_csv(init_history, index_label='index')

    # Perform optimization
    optimizer.initialize(init_history)
    optimizer.maximize(n_iter=num_iter_seq*par, parallelism_level=par,
                       timeout=timeout)
    logger.info("Run with RNG seed %d completed", rng)


# Loop over paralellism levels and RNG seeds
logger.info("Starting experiment campaign with parallelism = %d, n_init = %d",
            parallelism, n_init)
for par in all_parallelism_levels:
  logger.info("Starting experiments with par = %d", par)
  # Initialize RNG seeds
  rng_seeds = [root_rng_seed+i for i in range(num_runs)]
  logger.info("List of main RNG seeds: %s", rng_seeds)
  # Further seeds for independent sequential runs
  if par == 1 and parallelism > 1:
    for r in list(rng_seeds):
      group_seeds = [r] + [10*r+i for i in range(parallelism-1)]
      logger.info("List of RNG seeds of batch: %s", group_seeds)
      # Run such experiments in parallel
      with Pool(pool_seq_parallelism) as pool:
        pool.map(run_experiment, group_seeds)
      logger.info("Parallel batch completed")
  else:
    for rng in rng_seeds:
      run_experiment(rng)
  logger.info("Experiments with par = %d completed", par)

logger.info("Experiment campaign completed")
logger.info("Bye!")
