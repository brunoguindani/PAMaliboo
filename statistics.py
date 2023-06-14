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

import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd


# Campaign parameters
parallelism_levels = [1, 4]
indep_seq_runs = 4
num_runs = 10
root_rng_seed = 20230524
root_output_folder = 'outputs_ligen'
opt_constraints = {'RMSD_0.75': (0, 2)}

# Find real optimum
df_truth = pd.read_csv(os.path.join('resources', 'ligen',
                                    'ligen_red_table.csv'))
df_truth['target'] = -df_truth['RMSD_0.75'] ** 3 * df_truth['TIME_TOTAL']
df_truth.sort_values(by='target', inplace=True, ascending=False)
best = df_truth.iloc[0]

# Initialize main RNG seeds
main_rng_seeds = [root_rng_seed+i for i in range(num_runs)]
rng_to_par_to_results = {m: {} for m in main_rng_seeds}

for main_rng in main_rng_seeds:
  par_to_results = {p: {} for p in parallelism_levels}
  for par in parallelism_levels:
    # For independent sequential experiments, each main RNG seed has a *group*
    # of `indep_seq_runs` linked RNG seeds, which includes the main seed
    # itself. Otherwise, the group reduces to just the main seed
    group_seeds = [main_rng]
    if par == 1 and indep_seq_runs > 1:
      other_seeds = [10*main_rng+i for i in range(indep_seq_runs-1)]
      group_seeds.extend(other_seeds)
    print(f">>> par = {par}, main_rng = {main_rng}, -> {group_seeds}")

    # Initialize results dictionaries for this group
    res_dic      = dict.fromkeys(group_seeds, None)
    n_unfeas_dic = dict.fromkeys(group_seeds, None)
    avg_reg_dic  = dict.fromkeys(group_seeds, None)
    avg_mape_dic = dict.fromkeys(group_seeds, None)

    # Loop over individual RNG seeds in this group
    for rng in group_seeds:
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
      res['incumb'] = incumbents

      # Simple relative regret
      res['relreg'] = (res['incumb'] - best['target']) / best['target']

      # Remove initial points and compute global metrics
      noninit = (hist.index != -1)
      res = res.loc[noninit]
      n_unfeas = (~res['feas']).sum()
      avg_reg = res['relreg'].mean()

      # Add stuff to results dictionaries
      res_dic[rng] = res
      n_unfeas_dic[rng] = n_unfeas
      avg_reg_dic[rng] = avg_reg
      avg_mape_dic[rng] = pd.read_csv(os.path.join(output_folder,
                                                   'info.csv'))['train_MAPE']

    # Concatenate results horizontally and compute best incumbent/regret across
    # seeds, for each iteration (row)
    res_concat = pd.concat(list(res_dic.values()), axis=1)
    best_incumb = pd.DataFrame(res_concat['incumb']) \
                    .max(axis=1).rename('incumb', inplace=True)
    best_relreg = pd.DataFrame(res_concat['relreg']) \
                    .min(axis=1).rename('relreg', inplace=True)
    # Combine results into single DataFrame
    best_combined = pd.concat((best_incumb, best_relreg), axis=1)
    avg_mape = pd.concat(avg_mape_dic.values(), axis=1).mean(axis=1)

    # Compute other group metrics
    group_n_unfeas = np.mean(list(n_unfeas_dic.values()))
    group_avg_reg = np.mean(list(avg_reg_dic.values()))

    # Print stuff
    # print("Group average metrics:")
    # print("Unfeasible =", group_n_unfeas)
    # print("Average regret =", group_avg_reg)
    # print("Iteration-related metrics:")
    # print(best_combined)
    # print("\n")

    par_to_results[par]['n_unfeas'] = group_n_unfeas
    par_to_results[par]['avg_reg'] = group_avg_reg
    par_to_results[par]['iterations'] = best_combined
    par_to_results[par]['avg_mape'] = avg_mape

    rng_to_par_to_results[main_rng] = par_to_results

  # For each main RNG seeed, print and plot stuff
  print(f"For main RNG seed {main_rng}:")
  fig, ax = plt.subplots(2, 1, figsize=(5, 8))
  for par in parallelism_levels:
    print(f"par = {par}: n_unfeas = {par_to_results[par]['n_unfeas']}, "
          f"avg_reg = {par_to_results[par]['avg_reg']}")
    ax[0].plot(par_to_results[par]['iterations']['relreg'], marker='o',
                                                            label=str(par))
    ax[1].plot(par_to_results[par]['avg_mape'], marker='o', label=str(par))
  ax[0].axhline(0, c='lightgreen', ls='--', label='ground truth')
  ax[0].set_ylim(-0.01, 1.0)
  ax[0].set_title("Relative regret")
  ax[0].legend()

  ax[1].set_ylim(-0.01, 0.1)
  ax[1].set_title("Training MAPE")
  ax[1].legend()

  plot_file = os.path.join(root_output_folder,
                           f'par_vs_{indep_seq_runs}_{main_rng}.png')
  fig.savefig(plot_file, bbox_inches='tight', dpi=300)
  print()

print("Global metrics:")
for par in parallelism_levels:
  nums_unfeas = [ rng_to_par_to_results[r][par]['n_unfeas']
                  for r in main_rng_seeds]
  avg_regs = [ rng_to_par_to_results[r][par]['avg_reg']
               for r in main_rng_seeds]
  print(f"par = {par}: n_unfeas = {np.mean(nums_unfeas)}, "
        f"avg_reg = {np.mean(avg_regs)}")
