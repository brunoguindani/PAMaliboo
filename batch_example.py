import logging
import os
import pandas as pd
import sys

from pamaliboo.batch import BatchExecutor
from pamaliboo.jobs import HyperqueueJobSubmitter
from pamaliboo.objectives import DummyObjective


# Initialize configuration dataframe
columns = ['x1', 'x2']
configs = [[3.0, 1.0],
           [2.0, 4.0]]
df = pd.DataFrame(data=configs, columns=columns)

# Initialize library objects
logging.basicConfig(level=logging.DEBUG)
job_submitter = HyperqueueJobSubmitter('outputs')
obj = DummyObjective()
batch_ex = BatchExecutor(job_submitter, obj)

# Perform batch execution
res = batch_ex.execute(df, timeout=0.5)

# Change index to -1's
new_idx = pd.Index([-1]*res.shape[0])
res.set_index(new_idx, inplace=True)
res.to_csv(os.path.join('resources', 'dummy_initial.csv'), index_label='index')
