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
import pandas as pd


class FileDataFrame:
  """
  File-based database.

  This class represents a tabular data structure, but all its contents are
  saved at all times to a text file, called the database. The path to the
  database must be provided to the class constructor. Users can exploit the
  class interface to add and remove individual points to the database. While
  performing these operations, the database is temporarily treated as a
  pandas.DataFrame, which is saved to the `self.df` member. Recall at all times
  the core philosophy of this class: only the content of the database file
  matters!
  """
  index_name = 'index'

  def __init__(self, file_path: str, *args, **kwargs):
    self.logger = logging.getLogger(__name__)
    self.file_path = file_path

    if os.path.exists(self.file_path):
      self.logger.debug("%s exists, reading dataframe from file ignoring args",
                        self.file_path)
      self._read()
    else:
      self.logger.debug("%s does not exist, initializing new dataframe with "
                        "args=%s, kwargs=%s", self.file_path, args, kwargs)
      self.df = pd.DataFrame(*args, **kwargs)
      self._save()


  def __len__(self) -> int:
    self._read()
    length = self.df.shape[0]
    self._save()
    return length

  def __contains__(self, idx) -> bool:
    return idx in self.df.index


  def _read(self) -> None:
    """Read the DataFrame from file. Internal use only!"""
    self.df = pd.read_csv(self.file_path, index_col=self.index_name)


  def _save(self) -> None:
    """Save the DataFrame to file. Internal use only!"""
    self.df.to_csv(self.file_path, index_label=self.index_name)


  def get_df(self) -> pd.DataFrame:
    """Return DataFrame object"""
    self._read()
    return self.df


  def add_row(self, index: int, row: np.ndarray) -> None:
    """Add new row at the given index value"""
    self._read()
    self.df.loc[index] = row
    self.logger.debug("New row with index %d added to %s: %s", index,
                      self.file_path, self.df.loc[index].to_dict())
    self._save()


  def remove_row(self, index: int) -> None:
    """Remove row at the given index value"""
    self._read()
    self.logger.debug("Removing row with index %d from %s: %s", index,
                      self.file_path, self.df.loc[index].to_dict())
    self.df.drop(index, axis=0, inplace=True)
    self._save()
