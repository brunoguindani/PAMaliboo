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
import os
import pandas as pd


class FileDataFrame:
  index_name = 'index'

  def __init__(self, file_path: str, *args, **kwargs):
    self.file_path = file_path

    if os.path.exists(self.file_path):
      self.read()
    else:
      self.df = pd.DataFrame(*args, **kwargs)
      self.save()


  def read(self) -> None:
    self.df = pd.read_csv(self.file_path, index_col=self.index_name)


  def save(self) -> None:
    self.df.to_csv(self.file_path, index_label=self.index_name)


  def get_df(self) -> pd.DataFrame:
    self.read()
    return self.df


  def add_row(self, index: int, row: np.ndarray) -> None:
    self.read()
    self.df.loc[index] = row
    self.save()


  def remove_row(self, index: int) -> None:
    self.read()
    self.df.drop(index, axis=0, inplace=True)
    self.save()
