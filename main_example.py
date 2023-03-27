import numpy as np
import os
from shutil import copyfile

from pamaliboo.acquisitions import UpperConfidenceBound, ExpectedImprovement
from pamaliboo.gaussian_process import DatabaseGaussianProcessRegressor as DGPR
from pamaliboo.jobs import HyperqueueJobSubmitter
from pamaliboo.objectives import DummyObjective
from pamaliboo.optimizer import Optimizer


output_folder = 'outputs'
database = 'dummy.local.csv'
np.random.seed(42)

# Temporary code to take initialization values and remove temp files
database_path = os.path.join(output_folder, database)
if os.path.exists(database_path):
  os.remove(database_path)
copyfile('outputs/dummy_initial.local.csv', database_path)
queue_path = os.path.join(output_folder, 'jobs_queue.csv')
if os.path.exists(queue_path):
  os.remove(queue_path)
real_points_path = os.path.join(output_folder, 'real_points.csv')
if os.path.exists(real_points_path):
  os.remove(real_points_path)

# Initialize library objects
acq = UpperConfidenceBound()
bounds = {'x1': (-20, 20), 'x2': (-20, 20)}
gp = DGPR(database_path, feature_names=['f1', 'f2'])
gp.fit()
job_submitter = HyperqueueJobSubmitter(output_folder)
obj = DummyObjective()
optimizer = Optimizer(acq, bounds, gp, job_submitter, obj, output_folder)
optimizer.maximize(n_iter=5, parallelism_level=2, timeout=5)
