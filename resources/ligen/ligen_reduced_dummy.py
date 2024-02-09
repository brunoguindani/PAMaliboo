import numpy as np
import pandas as pd
from time import sleep
import sys

# Input vector
x = np.array([ int(_) for _ in sys.argv[1:] ])
# Domain
df = pd.read_csv('resources/ligen/ligen_red_table.csv')
mat = df.values[:, 0:3]
# Point corresponding to x
idx = np.where((mat == x).all(axis=1))[0][0]
rmsd, time = df.iloc[idx][['RMSD_0.75', 'TIME_TOTAL']]
sleep(2)
print(rmsd, time)
