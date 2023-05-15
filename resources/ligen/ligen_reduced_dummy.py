import numpy as np
import pandas as pd
from time import sleep
import sys

# Input vector
x = np.array([ int(_) for _ in sys.argv[1:] ])
# Domain
df = pd.read_csv('resources/ligen/ligen_casf-2016.csv')
mat = df.values[:, 0:3]
# Points corresponding to x
idxs = np.where((mat == x).all(axis=1))[0]
output = df.loc[idxs, 'RMSD'].quantile(0.75)
sleep(2)
print(output)
