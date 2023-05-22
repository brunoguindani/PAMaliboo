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

from pamaliboo.acquisitions import UpperConfidenceBound, ExpectedImprovement
from pamaliboo.batch import BatchExecutor
from pamaliboo.gaussian_process import DatabaseGaussianProcessRegressor as DGPR
from pamaliboo.jobs import HyperqueueJobSubmitter
from pamaliboo.objectives import LigenReducedDummyObjective
from pamaliboo.optimizer import Optimizer


output_folder = 'outputs'
os.makedirs(output_folder, exist_ok=True)
gp_database = os.path.join(output_folder, 'gp_database.csv')
init_history = os.path.join(output_folder, 'ligen_dummy_initial.csv')
domain = os.path.join('resources', 'ligen', 'ligen_red_domain.csv')
timeout = 3
rng_seed = 42
# Initial points
features = ['ALIGN', 'OPT' ,'REPS']
configs = [[32,  8, 1],
           [ 8, 72, 4],
          ]
df_init = pd.DataFrame(data=configs, columns=features)

# Initialize and set relevant stuff
debug = True if '-d' in sys.argv or '--debug' in sys.argv else False
logging.basicConfig(level=logging.DEBUG if debug else logging.INFO)
np.random.seed(rng_seed)

# Initialize library objects
job_submitter = HyperqueueJobSubmitter(output_folder)
opt_bounds = {'ALIGN': (8, 72.01), 'OPT': (8, 72.01) ,'REPS': (1, 5.01)}
# model = Ridge()
acq = UpperConfidenceBound(maximize_n_warmup=10, maximize_n_iter=100)
kernel = Matern(nu=2.5)
gp = DGPR(gp_database, feature_names=features, kernel=kernel)
# obj = DummyObjective()
obj = LigenReducedDummyObjective(domain_file=domain)
optimizer = Optimizer(acq, opt_bounds, gp, job_submitter, obj, output_folder)

# Run initial points
batch_ex = BatchExecutor(job_submitter, obj)
res = batch_ex.execute(df_init, timeout=timeout)
new_idx = pd.Index([-1]*res.shape[0])
res.set_index(new_idx, inplace=True)
res.to_csv(init_history, index_label='index')

# Perform optimization
optimizer.initialize(init_history)
optimizer.maximize(n_iter=20, parallelism_level=2, timeout=timeout)
