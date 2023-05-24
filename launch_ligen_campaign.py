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
from sklearn.gaussian_process.kernels import Matern, WhiteKernel
from sklearn.linear_model import Ridge
import sys

from pamaliboo.acquisitions import UpperConfidenceBound, ExpectedImprovement, \
                                   ExpectedImprovementMachineLearning as EIML
from pamaliboo.batch import BatchExecutor
from pamaliboo.gaussian_process import DatabaseGaussianProcessRegressor as DGPR
from pamaliboo.jobs import HyperqueueJobSubmitter
from pamaliboo.objectives import LigenReducedDummyObjective
from pamaliboo.optimizer import Optimizer


# Campaign parameters
parallelism_levels = [1, 4]
num_runs = 10
num_iter = 30
root_rng_seed = 20230524
root_output_folder = 'outputs_ligen'
ml_models = [Ridge()]

# Other parameters
opt_bounds = {'ALIGN': (8, 72.01), 'OPT': (8, 72.01) ,'REPS': (1, 5.01)}
opt_constraints = {'RMSD_0.75': (0, 2)}
features = list(opt_bounds.keys())
domain = os.path.join('resources', 'ligen', 'ligen_red_domain.csv')
timeout = 3

# Initialize and set relevant stuff
rng_seeds = [root_rng_seed+i for i in range(num_runs)]
debug = True if '-d' in sys.argv or '--debug' in sys.argv else False
logging.basicConfig(level=logging.DEBUG if debug else logging.INFO)

# Initialize constant library objects
acq = EIML(constraints=opt_constraints, models=ml_models,
           pickle_folder=None, maximize_n_warmup=10, maximize_n_iter=100)
kernel = Matern(nu=2.5)
obj = LigenReducedDummyObjective(domain_file=domain)


for par in parallelism_levels:
  for rng in rng_seeds:
    output_folder = os.path.join(root_output_folder, f'par_{par}',
                                                     f'rng_{rng}')
    os.makedirs(output_folder, exist_ok=True)
    gp_database = os.path.join(output_folder, 'gp_database.csv')

    # # TODO get df_init
    # np.random.seed(rng)
    # init_history = 'TODO'

    # Initialize library objects
    job_submitter = HyperqueueJobSubmitter(output_folder)
    batch_ex = BatchExecutor(job_submitter, obj)
    gp = DGPR(gp_database, feature_names=features, kernel=kernel,
                           normalize_y=True)
    optimizer = Optimizer(acq, opt_bounds, gp, job_submitter, obj,
                          output_folder)

    continue  # TODO

    # Run initial points
    res = batch_ex.execute(df_init, timeout=timeout)
    new_idx = pd.Index([-1]*res.shape[0])
    res.set_index(new_idx, inplace=True)
    res.to_csv(init_history, index_label='index')

    # Perform optimization
    optimizer.initialize(init_history)
    optimizer.maximize(n_iter=num_iter, parallelism_level=par, timeout=timeout)