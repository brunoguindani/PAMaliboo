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


def df_to_Xy(df: pd.DataFrame, y_column: str) -> tuple[np.ndarray, np.ndarray]:
  y = df[y_column].values
  df.drop(y_column, axis=1, inplace=True)
  X = df.values
  return X, y


def dict_to_array(dic: dict[str: tuple[float]]) -> np.ndarray:
  return np.array(list(dic.values()))


def join_Xy(X: np.ndarray, y: float) -> np.ndarray:
  return np.hstack((X, [y]))
