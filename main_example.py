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
from sklearn.gaussian_process.kernels import Matern
from sklearn.linear_model import Ridge
import sys

from pamaliboo.acquisitions import ExpectedImprovementMachineLearning as EIML
from pamaliboo.gaussian_process import DatabaseGaussianProcessRegressor as DGPR
from pamaliboo.jobs import HyperqueueJobSubmitter
from pamaliboo.objectives import DummyObjective
from pamaliboo.optimizer import Optimizer


output_folder = 'outputs'
gp_database = os.path.join(output_folder, 'gp_database.csv')
init_history = os.path.join('resources', 'dummy_initial.csv')
domain = os.path.join('resources', 'dummy_domain.csv')
rng_seed = 42


# Initialize and set relevant stuff
debug = True if '-d' in sys.argv or '--debug' in sys.argv else False
logging.basicConfig(level=logging.DEBUG if debug else logging.INFO)
np.random.seed(rng_seed)


# Initialize library objects
job_submitter = HyperqueueJobSubmitter(output_folder)
opt_bounds = {'x1': (0, 5), 'x2': (0, 5)}
features = list(opt_bounds.keys())
constraints = {'result': (2, 6), 'result^2': (4, 36)}
model = Ridge()
acq = EIML(maximize_n_warmup=10, maximize_n_iter=100, constraints=constraints,
           models=[model, model], pickle_folder=output_folder)
kernel = Matern(nu=2.5)
gp = DGPR(gp_database, feature_names=features, kernel=kernel,
          normalize_y=False)
# obj = DummyObjective()
obj = DummyObjective(domain_file=domain)
optimizer = Optimizer(acq, opt_bounds, gp, job_submitter, obj, output_folder)
optimizer.initialize(init_history)
optimizer.maximize(n_iter=10, parallelism_level=2, timeout=3)
