import numpy as np
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pamaliboo.gaussian_process import DatabaseGaussianProcessRegressor as DGPR
from pamaliboo.acquisitions import (UpperConfidenceBound as UCB,
                                    ExpectedImprovement as EI)


def test_acq_gp():
  # Test GP
  database = 'tests/temp.local.csv'
  np.random.seed(42)
  x0 = np.array([[3,7]])
  gp = DGPR(database, feature_names=['f1', 'f2'])
  X = np.array([[2, 5],
                [4, 10]])
  y = np.array([20, 50])
  gp.add_point(11, X[0,:], y[0])
  gp.add_point(12, X[1,:], y[1])
  gp.remove_point(12)
  gp.add_point(12, X[1,:], y[1])
  print(gp.database.get_df())
  gp.fit()
  pred = gp.predict(x0)
  print("Prediction on x0 =", pred)

  # Test acquisition
  bounds = {'x1': (1, 5), 'x2': (2, 12)}
  acq = UCB(kappa=2.5, maximize_n_warmup=10, maximize_n_iter=100)
  #acq = EI(xi=0, maximize_n_warmup=10, maximize_n_iter=100)
  acq.update_state(gp, 3)
  acq_val = acq.evaluate(x0)
  print("acq(x0) =", acq_val)
  print("Maximizing acq...")
  x_best, acq_best = acq.maximize(bounds)
  print("x =", x_best, ", acq(x) =", acq_best)
