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
experiment_kind = 'red'
use_relative = True
use_incumbents = True
parallelism_levels = [1, 4]
indep_seq_runs = 4
num_runs = 10
root_rng_seed = 20230524
opt_constraints = {'RMSD_0.75': (0, 2)}
root_output_folder = f'outputs_ligen_{experiment_kind}'

# Find real optimum
df_truth = pd.read_csv(os.path.join('resources', 'ligen',
                                   f'ligen_{experiment_kind}_table.csv'))
df_truth['target'] = -df_truth['RMSD_0.75'] ** 3 * df_truth['TIME_TOTAL']
df_truth.sort_values(by='target', inplace=True, ascending=False)
best = df_truth.iloc[0]
# print(best)

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
    avg_dist_dic = dict.fromkeys(group_seeds, None)
    avg_mape_dic = dict.fromkeys(group_seeds, None)
    time_dist_dic = dict.fromkeys(group_seeds, None)
    time_feas_dic = {}

    # Loop over individual RNG seeds in this group
    for rng in group_seeds:
      # Get history dataframe for this experiment
      output_folder = os.path.join(root_output_folder, f'par_{par}',
                                                       f'rng_{rng}')
      hist = pd.read_csv(os.path.join(output_folder, 'history.csv'),
                         index_col='index')
      info = pd.read_csv(os.path.join(output_folder, 'info.csv'),
                         index_col='index')
      res = pd.DataFrame(index=hist.index)

      # Feasible observations with respect to the constraints
      res['feas'] = True
      for key, (lb, ub) in opt_constraints.items():
        res['feas'] = res['feas'] & (lb <= hist[key]) & (hist[key] <= ub)

      # Collect *feasible* points at each iterations
      points = []
      if use_incumbents:
        # collect incumbents at each iteration
        curr = None
        for i in range(hist.shape[0]):
          if res['feas'].iloc[i] and (curr is None
                                      or hist['target'].iloc[i] > curr):
            curr = hist['target'].iloc[i]
          points.append(curr)
      else:  # if not use_incumbents
        # collect the points evaluated at each iteration
        for i in range(hist.shape[0]):
          if res['feas'].iloc[i]:
            points.append(hist['target'].iloc[i])
          else:
            points.append(None)
      # Save points to results dataframe
      res['points'] = points

      # Compute distance from ground truth, either:
      if use_relative:
        # ...simple relative regret
        res['dist'] = (res['points'] - best['target']) / best['target']
      else:
        # ...or target value
        res['dist'] = -res['points']

      # Remove initial points and compute global metrics
      noninit = (hist.index != -1)
      res = res.loc[noninit]
      n_unfeas = (~res['feas']).sum()
      avg_dist = res['dist'].mean()

      # Compute optimizer times on the time grid
      discrete_times = info['optimizer_time']
      dists = res['dist']
      # Initialize vector of time instants
      time_grid = np.arange(0, discrete_times.iloc[-1]+1.0, 1.0)
      time_dist = pd.Series(index=time_grid)
      for i in range(len(discrete_times)-1):
        time_dist[discrete_times[i]:discrete_times[i+1]] = dists[i]
      time_dist[discrete_times.iloc[-1]:] = dists.iloc[-1]

      # Collect time -> feasibility
      for t, feas in zip(discrete_times, res['feas']):
        time_feas_dic[t] = feas

      # Add stuff to results dictionaries
      res_dic[rng] = res
      n_unfeas_dic[rng] = n_unfeas
      avg_dist_dic[rng] = avg_dist
      avg_mape_dic[rng] = info['train_MAPE']
      time_dist_dic[rng] = time_dist

    # Concatenate results horizontally and compute best points and distance
    # across seeds, for each iteration (row)
    res_concat = pd.concat(list(res_dic.values()), axis=1)
    # Combine results into single DataFrame
    avg_mape = pd.concat(avg_mape_dic.values(), axis=1).mean(axis=1)
    best_time_dist = pd.concat(time_dist_dic.values(), axis=1).min(axis=1)
    # Stop at the earliest time at which a seed has finished
    end_time = np.min([d.index[-1] for d in time_dist_dic.values()])
    best_time_dist = best_time_dist.loc[:end_time]

    # Compute other group metrics
    group_n_unfeas = np.mean(list(n_unfeas_dic.values()))
    group_avg_dist = np.mean(list(avg_dist_dic.values()))

    par_to_results[par]['n_unfeas'] = group_n_unfeas
    par_to_results[par]['avg_dist'] = group_avg_dist
    par_to_results[par]['avg_mape'] = avg_mape
    par_to_results[par]['time_dist'] = best_time_dist
    par_to_results[par]['num_indep_runs'] = len(res_dic)
    par_to_results[par]['time_feas'] = time_feas_dic

    rng_to_par_to_results[main_rng] = par_to_results

  # For each main RNG seeed, print and plot stuff
  print(f"For main RNG seed {main_rng}:")
  fig, ax = plt.subplots(2, 1, figsize=(8, 8))
  for par in parallelism_levels:
    par_n_unf = par_to_results[par]['n_unfeas']
    num_indep_runs = par_to_results[par]['num_indep_runs']
    label = f"parallelism {par}, {num_indep_runs} instance(s)"
    print(f"par = {par}: n_unfeas = {par_n_unf}, "
          f"avg_dist = {par_to_results[par]['avg_dist']}")

    # First plot: regret and feasibility of executions
    times_dists = par_to_results[par]['time_dist']
    ax[0].plot(times_dists, lw=1, label=label+f" (unfeasible: {par_n_unf})")
    color = ax[0].lines[-1].get_color()
    # Loop on iterations (their optimizer time and feasibility)
    for t, feas in par_to_results[par]['time_feas'].items():
      # Compute approximation of optimizer time on the time grid
      if t > times_dists.index[-1]:
        time_approx = times_dists.index[-1]
      else:
        time_approx = times_dists.index[times_dists.index > t][0]
      # Plot iteration differently according to its feasibility
      facecolor = color if feas else 'none'
      ax[0].scatter(time_approx, times_dists[time_approx], marker='o', s=12,
                    edgecolor=color, facecolors=facecolor, linewidths=0.5)

    # Second plot: MAPE over iterations
    ax[1].plot(par_to_results[par]['avg_mape'], marker='o', label=label)
    ground = 0 if use_relative else -best['target']
  ax[0].axhline(ground, c='lightgreen', ls='--', label="ground truth",
                        zorder=-2)
  if use_relative:
    ax[0].set_ylim(-0.01, 0.8)
    title_distance = "Relative regret"
  else:
    floor = np.floor(-best['target'] / 10**3) * 10**3
    ax[0].set_ylim(floor, 2*floor)
    title_distance = "Target values"
  title_points = "incumbents" if use_incumbents else "points"
  ax[0].set_xlabel("time [s]")
  ax[0].grid(axis='y', alpha=0.4)
  ax[0].set_title(f"{title_distance} of {title_points}")
  ax[0].legend()
  ax[1].set_xlabel("iterations")
  ax[1].set_ylim(-0.01, 0.1)
  ax[1].grid(axis='y', alpha=0.4)
  ax[1].set_title("Training MAPE")
  ax[1].legend()
  fig.subplots_adjust(hspace=0.25)

  plot_file = os.path.join(root_output_folder,
                           f'par_vs_{indep_seq_runs}_{main_rng}.png')
  fig.savefig(plot_file, bbox_inches='tight', dpi=300)
  print()

fig, ax = plt.subplots(2, 1, figsize=(8, 8))
for par in parallelism_levels:
  df_dist = pd.concat([rng_to_par_to_results[r][par]['time_dist']
                       for r in main_rng_seeds], axis=1)
  df_dist = df_dist.fillna(method='ffill').mean(axis=1)
  df_mape = pd.concat([rng_to_par_to_results[r][par]['avg_mape']
                       for r in main_rng_seeds], axis=1).fillna(method='ffill')
  df_mape = df_mape.fillna(method='ffill').mean(axis=1)
  num_indep_runs = rng_to_par_to_results[main_rng_seeds[0]] \
                                        [par]['num_indep_runs']
  label = f"parallelism {par}, {num_indep_runs} instance(s)"
  ax[0].plot(df_dist, lw=1, label=label)
  ax[1].plot(df_mape, marker='o', label=label)

ax[0].axhline(ground, c='lightgreen', ls='--', label="ground truth",
                      zorder=-2)
if use_relative:
  ax[0].set_ylim(-0.01, 0.5)
  title_distance = "Relative regret"
else:
  floor = np.floor(-best['target'] / 10**3) * 10**3
  ax[0].set_ylim(floor, 2*floor)
  title_distance = "Target values"
title_points = "incumbents" if use_incumbents else "points"
ax[0].set_xlabel("time [s]")
ax[0].grid(axis='y', alpha=0.4)
ax[0].set_title(f"{title_distance} of {title_points}")
ax[0].legend()
ax[1].set_xlabel("iterations")
ax[1].set_ylim(-0.01, 0.1)
ax[1].grid(axis='y', alpha=0.4)
ax[1].set_title("Training MAPE")
ax[1].legend()
fig.subplots_adjust(hspace=0.25)
plot_file = os.path.join(root_output_folder,
                         f'par_vs_{indep_seq_runs}_all.png')
fig.savefig(plot_file, bbox_inches='tight', dpi=300)

print("Global metrics:")
for par in parallelism_levels:
  nums_unfeas = [ rng_to_par_to_results[r][par]['n_unfeas']
                  for r in main_rng_seeds]
  avg_dists = [ rng_to_par_to_results[r][par]['avg_dist']
               for r in main_rng_seeds]
  print(f"par = {par}: n_unfeas = {np.mean(nums_unfeas)}, "
        f"avg_dist = {np.mean(avg_dists)}")
