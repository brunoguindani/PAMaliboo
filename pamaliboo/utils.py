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
from typing import Dict, Tuple


def df_to_Xy(df: pd.DataFrame, y_column: str) -> Tuple[np.ndarray, np.ndarray]:
  """Separate DataFrame into a y column with the given name, and an X matrix"""
  y = df[y_column].values
  df.drop(y_column, axis=1, inplace=True)
  X = df.values
  return X, y


def dict_to_array(dic: Dict[str, Tuple[float]]) -> np.ndarray:
  """Transform a dictionary into a numpy array"""
  return np.array(list(dic.values()))


def join_Xy(X: np.ndarray, y: float) -> np.ndarray:
  """Join horizontally a row (or matrix) `X` and a value (or column) `y`"""
  return np.hstack((X, [y]))


def numpy_to_str(array: np.ndarray) -> str:
  """Convert a `numpy.array` into a list of numbers in string form"""
  return '/'.join([str(_) for _ in array])


def str_to_numpy(strg: str) -> np.ndarray:
  """Convert a list of numbers in string form, into a `numpy.array`"""
  return np.array([float(_) for _ in strg.split('/')])
