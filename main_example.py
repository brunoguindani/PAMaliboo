import logging
import numpy as np
import os
import pandas as pd
from shutil import copyfile
from sklearn.gaussian_process.kernels import Matern, WhiteKernel
from sklearn.linear_model import Ridge
import sys

from pamaliboo.acquisitions import UpperConfidenceBound, ExpectedImprovement, \
                                   ExpectedImprovementMachineLearning as EIML
from pamaliboo.gaussian_process import DatabaseGaussianProcessRegressor as DGPR
from pamaliboo.jobs import HyperqueueJobSubmitter
from pamaliboo.objectives import DummyObjective
from pamaliboo.optimizer import Optimizer


output_folder = 'outputs'
database = 'gp_database.csv'
init_history = os.path.join('resources', 'dummy_initial.csv')
domain = os.path.join('resources', 'dummy_domain.csv')
rng_seed = 42


# Initialize and set relevant stuff
debug = True if '-d' in sys.argv or '--debug' in sys.argv else False
logging.basicConfig(level=logging.DEBUG if debug else logging.INFO)
database_path = os.path.join(output_folder, database)
np.random.seed(rng_seed)


# Initialize library objects
job_submitter = HyperqueueJobSubmitter(output_folder)
constraints = {'result': (2, 6), 'result^2': (4, 36)}
model = Ridge()
acq = EIML(maximize_n_warmup=10, maximize_n_iter=100, constraints=constraints,
           models=[model, model])
kernel = Matern(nu=2.5)
gp = DGPR(database_path, feature_names=['f1', 'f2'], kernel=kernel)
# obj = DummyObjective()
obj = DummyObjective(domain_file=domain)
opt_bounds = {'x1': (0, 5), 'x2': (0, 5)}
optimizer = Optimizer(acq, opt_bounds, gp, job_submitter, obj, output_folder)
optimizer.initialize(init_history)
optimizer.maximize(n_iter=10, parallelism_level=2, timeout=3)
