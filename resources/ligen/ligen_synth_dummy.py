import numpy as np
import pandas as pd
from time import sleep
import sys

# Input vector
x = np.array([ int(_) for _ in sys.argv[1:] ])
# Domain
df = pd.read_csv('resources/ligen/ligen_synth_table.csv')
domain = df.iloc[:, 0:8]
# Find best approximation to x
norm = lambda y : np.linalg.norm(x-y)
idx = domain.apply(norm, axis=1).argmin()
# Print relevant metrics
rmsd, time = df.loc[idx, ['RMSD_0.75', 'TIME_TOTAL']]
sleep(time/500)
print(rmsd, time)
