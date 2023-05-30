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

import os
import pandas as pd


# Campaign parameters
parallelism_levels = [1, 2]
indep_seq_runs = 4
num_runs = 3
root_rng_seed = 20230524
root_output_folder = 'outputs_ligen'
opt_constraints = {'RMSD_0.75': (0, 2)}

# Find real optimum
df_truth = pd.read_csv(os.path.join('resources', 'ligen',
                                    'ligen_red_table.csv'))
df_truth['target'] = -df_truth['RMSD_0.75'] ** 3 * df_truth['TIME_TOTAL']
df_truth.sort_values(by='target', inplace=True, ascending=False)
best = df_truth.iloc[0]

# Loop over paralellism levels and RNG seeds
for par in parallelism_levels:
  # Initialize RNG seeds
  rng_seeds = [root_rng_seed+i for i in range(num_runs)]
  # Further seeds for independent sequential runs
  if par == 1 and indep_seq_runs > 1:
    for r in list(rng_seeds):
      other_seeds = [10*r+i for i in range(indep_seq_runs-1)]
      rng_seeds.extend(other_seeds)

  for rng in rng_seeds:
    print(f">>> par = {par}, rng = {rng}")
    # Get history dataframe for this experiment
    output_folder = os.path.join(root_output_folder, f'par_{par}',
                                                     f'rng_{rng}')
    hist = pd.read_csv(os.path.join(output_folder, 'history.csv'),
                       index_col='index')
    res = pd.DataFrame(index=hist.index)

    # Feasible observations with respect to the constraints
    res['feas'] = True
    for key, (lb, ub) in opt_constraints.items():
      res['feas'] = res['feas'] & (lb <= hist[key]) & (hist[key] <= ub)

    # Feasible incumbents at each iteration
    incumbents = []
    curr_inc = None
    for i in range(hist.shape[0]):
      if res['feas'].iloc[i] and (curr_inc is None
                                  or hist['target'].iloc[i] > curr_inc):
        curr_inc = hist['target'].iloc[i]
      incumbents.append(curr_inc)
    res['incum'] = incumbents

    # Simple relative regret
    res['relreg'] = (res['incum'] - best['target']) / best['target']

    # Print results and compute global metrics by removing initial points
    noninit = (hist.index != -1)
    res = res.loc[noninit]
    print(res)
    print("Unfeasible =", (~res['feas']).sum())
    print("Average relative regret =", res['relreg'].mean())
    print()
