import logging
import numpy as np
import os
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
database = 'dummy.local.csv'
additional_info = 'dummy_add_info.csv'
np.random.seed(42)
debug = True if '-d' in sys.argv or '--debug' in sys.argv else False

# Temporary code to take initialization values and remove temp files
os.makedirs(output_folder, exist_ok=True)
database_path = os.path.join(output_folder, database)
additional_info_path = os.path.join(output_folder, additional_info)
if not os.path.exists(database_path):
  copyfile('resources/dummy_initial.csv', database_path)
  copyfile('resources/dummy_add_info_initial.csv', additional_info_path)

# Set logging level
logging.basicConfig(level=logging.DEBUG if debug else logging.INFO)

# Initialize library objects
constraints = {'result': (2, 6), 'result^2': (4, 36)}
model = Ridge()
acq = EIML(maximize_n_warmup=10, maximize_n_iter=100, models=[model, model],
           constraints=constraints, additional_info_path=additional_info_path)
bounds = {'x1': (0, 5), 'x2': (0, 5)}
kernel = Matern(nu=2.5)  # + WhiteKernel(noise_level_bounds=(0.2, 2))
gp = DGPR(database_path, feature_names=['f1', 'f2'], kernel=kernel)
gp.fit()
job_submitter = HyperqueueJobSubmitter(output_folder)
# obj = DummyObjective()
obj = DummyObjective(domain_file='resources/dummy_domain.csv')
optimizer = Optimizer(acq, bounds, gp, job_submitter, obj, output_folder)
optimizer.maximize(n_iter=10, parallelism_level=2, timeout=3)
