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
from pandas.errors import EmptyDataError
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Kernel, Matern, WhiteKernel
from typing import List, Optional, Tuple, Union
import warnings

from .dataframe import FileDataFrame
from .utils import df_to_Xy, join_Xy


class DatabaseGaussianProcessRegressor(GaussianProcessRegressor):
  """
  Database-based Gaussian Processs regressor.

  It is a wrapper to scikit-learn's GaussianProcessRegressor, but all training
  data is always stored on disk. This is achieved by the use of the
  FileDataFrame class. Users can exploit the interface of this class to add or
  remove individual points to the database.
  """
  default_kernel = Matern(nu=2.5)
  index_column = 'index'
  target_column = 'target'

  def __init__(self, database: str, feature_names: List[str],
               kernel: Kernel = default_kernel, normalize_y: bool = True):
    """
    Parameters
    ----------
    `database`: path to the database file (either existing or to-be-created)
    `feature_names`: column names to be used in the database
    `kernel`: kernel object for the Gaussian Process prior
    `normalize_y`: normalize y values? (if data has nonzero mean, leave True)
    """
    self.logger = logging.getLogger(__name__)
    self.logger.debug("Initializing DGPR with database=%s, feature_names=%s, "
                      "kernel=%s", database, feature_names, kernel)

    super().__init__(kernel=kernel, normalize_y=normalize_y)

    self.database_path = database
    self.feature_names = feature_names

    self.database = FileDataFrame(self.database_path,
                                  columns=self.database_columns)


  @property
  def database_columns(self) -> List[str]:
    """Get the names of the features and of the target column"""
    return self.feature_names + [self.target_column]


  def read_database(self) -> None:
    """Update `X_train_` and `y_train_` members with data from the database"""
    df = self.database.get_df()
    self.X_train_, self.y_train_ = df_to_Xy(df, self.target_column)
    self.logger.debug("Setting X_train_ (shape %s) and y_train (shape %s)",
                      self.X_train_.shape, self.y_train_.shape)


  def add_point(self, index: int, X: np.ndarray, y: float) -> None:
    """Update database with a new point having data `X` and target value `y`"""
    row = join_Xy(X, y)
    self.logger.debug("Adding point %s with index %d", row, index)
    self.database.add_row(index, row)


  def get_point(self, index: int) -> np.ndarray:
    """Return point with the given index from database"""
    db = self.database.get_df()
    if index in db.index:
      return db.loc[index].to_numpy()
    else:
      raise IndexError(f"Point with index {index} not found")


  def remove_point(self, index: int) -> None:
    """Update database by removing the point with the given index"""
    self.logger.debug("Removing point with index %d", index)
    self.database.remove_row(index)


  def fit(self, X: Optional[np.ndarray] = None,
                y: Optional[np.ndarray] = None) -> GaussianProcessRegressor:
    """
    Fit Gaussian Process regression model.

    This overrides the behavior of the base object, so as to fail when passing
    argument `X` or `y`. The training data is instead drawn from the database.
    Therefore, the correct way to call this method is `fit()`.
    """
    if X is not None or y is not None:
      raise NotImplementedError("fit(X, y) cannot be used within this class. "
                                "Please use the database API to add training "
                                "data, then call fit() without arguments.")
    self.read_database()
    with warnings.catch_warnings():
      warnings.simplefilter('ignore')
      return super().fit(self.X_train_, self.y_train_)


  def predict(self, X: np.ndarray, return_std: bool = False) \
              -> Union[np.ndarray, Tuple[np.ndarray]]:
    """Wrapper for `predict()` that filters unwanted warnings."""
    if len(X.shape) == 1:
      X = X.reshape(1, -1)
    with warnings.catch_warnings():
      warnings.simplefilter('ignore')
      return super().predict(X, return_std=return_std)
