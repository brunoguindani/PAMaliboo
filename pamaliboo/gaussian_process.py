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

import numpy as np
import pandas as pd
from pandas.errors import EmptyDataError
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Kernel, Matern, WhiteKernel
from typing import Optional, Union

from .utils import df_to_Xy


class DatabaseGaussianProcessRegressor(GaussianProcessRegressor):
  """
  Database-based Gaussian Processs regressor.

  It is a wrapper to scikit-learn's GaussianProcessRegressor, but all training
  data is always stored on disk, in a .csv file called the database. The path
  to such file is fixed when initializing the object. Users can exploit the
  class interface to add or remove individual points to the database. While
  performing these operations, the database is temporarily treated as a
  pandas.DataFrame. Please recall at all times the core philosophy of this
  class: only the database on disk matters, not the contents of this class!
  """
  default_kernel = Matern(nu=2.5) + WhiteKernel()
  index_column = 'index'
  target_column = 'target'

  def __init__(self, database: str, feature_names: list[str],
               kernel: Kernel = default_kernel, *args, **kwargs):
    """
    Parameters
    ----------
    database: the path to the database file (either existing or to-be-created)
    feature_names: column names to be used in the database
    kernel: kernel object for the Gaussian Process prior
    """
    super().__init__(kernel=kernel, *args, **kwargs)

    self.database_path = database
    self.feature_names = feature_names

    starting_db = self.get_database()
    self._save_database(starting_db)


  @property
  def database_columns(self):
    return self.feature_names + [self.target_column]
  

  def get_database(self) -> pd.DataFrame:
    """
    Get database from disk, or empty DataFrame if the file is empty
    """
    try:
      return pd.read_csv(self.database_path, index_col=self.index_column)
    except EmptyDataError:
      return pd.DataFrame(columns=self.database_columns)


  def read_database(self) -> None:
    """
    Update the X_train_ and y_train_ members with data from the database
    """
    db = self.get_database()
    self.X_train_, self.y_train_ = df_to_Xy(db, self.target_column)


  def _save_database(self, db: pd.DataFrame) -> None:
    """Save given DataFrame to file. Internal use only!"""
    db.to_csv(self.database_path, index_label=self.index_column)


  def add_point_to_database(self, index: int, X: np.ndarray, y: float) -> None:
    """
    Update database with a new point having data X and target value y
    """
    db = self.get_database()
    row = np.hstack((X, [y]))
    print(row)
    db.loc[index] = row
    self._save_database(db)


  def remove_point_from_database(self, index: int) -> None:
    """
    Update database by removing the point with the given index
    """
    db = self.get_database()
    db.drop(index, axis=0, inplace=True)
    self._save_database(db)


  def fit(self, X: Optional[np.ndarray] = None,
                y: Optional[np.ndarray] = None) -> GaussianProcessRegressor:
    """
    Fit Gaussian Process regression model.

    This overrides the behavior of the base object, so as to fail when passing
    argument X or y. The training data is instead drawn from the database.
    Therefore, the correct way to call this method is: fit()
    """
    if X is not None or y is not None:
      raise NotImplementedError("fit(X, y) cannot be used within this class. "
                                "Please use the database API to add training "
                                "data, then call fit() without arguments.")
    self.read_database()
    return super().fit(self.X_train_, self.y_train_)


  def predict(self, *args, **kwargs) -> Union[np.ndarray, tuple[np.ndarray]]:
    """Wrapper for predict() that filters unwanted warnings."""
    with warnings.catch_warnings():
      warnings.simplefilter("ignore")
      return super().predict(*args, **kwargs)
