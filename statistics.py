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
rmsd_threshold = 2 if experiment_kind == 'red' else 2.1
opt_constraints = {'RMSD_0.75': (0, rmsd_threshold)}
root_output_folder = f'old/outputs_ligen_{experiment_kind}'

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
  # Initialize dict of global results for this main RNG seed
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

    # Initialize results dictionaries for this group: each entry links an RNG
    # seed in the group (i.e. an individual run) to a certain measurement:
    # SCALAR METRICS
    ## Number of unfeasible executions
    n_unfeas_dic = dict.fromkeys(group_seeds, None)
    ## Average distance (e.g. regret) on feasible points over all iterations
    avg_dist_dic = dict.fromkeys(group_seeds, None)
    ## Average distance (e.g. regret) on all points (both feasible and
    ## unfeasible) over all iterations
    avg_dist_fea_unf_dic = dict.fromkeys(group_seeds, None)
    # VECTORS OF METRICS
    ## MAPE on all iterations
    mape_dic = dict.fromkeys(group_seeds, None)
    ## Distance (e.g. regret) over time
    time_dist_dic = dict.fromkeys(group_seeds, None)
    ## Time elapsed at end of an execution -> feasibility of the execution
    time_feas_dic = {}

    # Loop over individual RNG seeds in this group
    for rng in group_seeds:
      # Get history dataframe for this experiment
      output_folder = os.path.join(root_output_folder, f'par_{par}',
                                                       f'rng_{rng}')
      hist = pd.read_csv(os.path.join(output_folder, 'history.csv'),
                         index_col='index')
      # Also get the recorded additional information
      info = pd.read_csv(os.path.join(output_folder, 'info.csv'),
                         index_col='index')

      # Check feasibility of observations with respect to the constraints
      feas = pd.Series(True, index=hist.index)
      for key, (lb, ub) in opt_constraints.items():
        feas = feas & (lb <= hist[key]) & (hist[key] <= ub)

      # Collect *feasible* points at each iterations, either:
      points = []
      if use_incumbents:
        # ...the incumbents at each iteration
        curr = np.nan
        for i in range(hist.shape[0]):
          if feas.iloc[i] and (curr is np.nan
                               or hist['target'].iloc[i] > curr):
            curr = hist['target'].iloc[i]
          points.append(curr)
      else:
        # ...or the points evaluated at each iteration
        for i in range(hist.shape[0]):
          if feas.iloc[i]:
            points.append(hist['target'].iloc[i])
          else:
            points.append(np.nan)
      points = pd.Series(points, index=hist.index)

      # Compute distance from ground truth, either:
      if use_relative:
        # ...simple relative regret
        dists = (points - best['target']) / best['target']
      else:
        # ...or target value
        dists = -points

      # Remove initial points
      noninit = (hist.index != -1)
      feas = feas.loc[noninit]
      points = points.loc[noninit]
      dists = dists.loc[noninit]

      # Distance considering both feasible and unfeasible points, either:
      targets = hist.loc[noninit, 'target']
      if use_relative:
        # ...simple relative regret
        dist_fea_unf = (targets - best['target']) / best['target']
      else:
        # ...or target value
        dist_fea_unf = -points
      avg_dist_fea_unf = dist_fea_unf.cummin().mean()

      # Get optimizer times for each evaluation
      discrete_times = info['optimizer_time']
      # Initialize vector of time instants
      delta = 1.0  # granularity
      time_grid = np.arange(0, discrete_times.iloc[-1]+delta, delta)
      time_dist = pd.Series(index=time_grid)
      # Collect current distance at each time instant in the grid
      for i in range(len(discrete_times)-1):
        time_dist[discrete_times[i]:discrete_times[i+1]] = dists[i]
      time_dist[discrete_times.iloc[-1]:] = dists.iloc[-1]

      # Collect time -> feasibility
      for t, feas_t in zip(discrete_times, feas):
        time_feas_dic[t] = feas_t

      # Add stuff to results dictionaries
      n_unfeas_dic[rng] = (~feas).sum()
      avg_dist_dic[rng] = dists.mean()
      avg_dist_fea_unf_dic[rng] = avg_dist_fea_unf
      mape_dic[rng] = info['train_MAPE']
      time_dist_dic[rng] = time_dist

    # Combine vectors of metrics into single DataFrames
    group_avg_mape = pd.concat(mape_dic.values(), axis=1).mean(axis=1)
    group_time_dist = pd.concat(time_dist_dic.values(), axis=1).min(axis=1)
    # Stop at the earliest time at which a seed has finished
    end_time = np.min([d.index[-1] for d in time_dist_dic.values()])
    group_time_dist = group_time_dist.loc[:end_time]

    # Collect scalar metrics
    par_to_results[par]['avg_n_unfeas'] = np.mean(list(n_unfeas_dic.values()))
    par_to_results[par]['avg_dist'] = np.mean(list(avg_dist_dic.values()))
    par_to_results[par]['avg_dist_fea_unf'] = \
                       np.mean(list(avg_dist_fea_unf_dic.values()))
    par_to_results[par]['num_indep_runs'] = len(group_seeds)
    # Collect combined vecctors of metrics
    par_to_results[par]['avg_mape'] = group_avg_mape
    par_to_results[par]['time_dist'] = group_time_dist
    par_to_results[par]['time_feas'] = time_feas_dic
    # Write metrics into the global results dict
    rng_to_par_to_results[main_rng] = par_to_results

  # For each main RNG seeed, print and plot stuff
  print(f"For main RNG seed {main_rng}:")
  fig, ax = plt.subplots(2, 1, figsize=(8, 8))
  for par in parallelism_levels:
    par_n_unf = par_to_results[par]['avg_n_unfeas']
    num_indep_runs = par_to_results[par]['num_indep_runs']
    label = f"parallelism {par}, {num_indep_runs} instance(s)"
    print(f"par = {par}: avg_n_unfeas = {par_n_unf}, "
          f"avg_dist = {par_to_results[par]['avg_dist']}")

    # First plot: distance and feasibility of executions over time
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
  # Other plot goodies
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
  # Save plot to file
  plot_file = os.path.join(root_output_folder,
                           f'par_vs_{indep_seq_runs}_{main_rng}.png')
  fig.savefig(plot_file, bbox_inches='tight', dpi=300)
  print()

# Make global plot
fig, ax = plt.subplots(2, 1, figsize=(8, 8))
for par in parallelism_levels:
  num_indep_runs = rng_to_par_to_results[main_rng_seeds[0]] \
                                        [par]['num_indep_runs']
  label = f"parallelism {par}, {num_indep_runs} instance(s)"
  # First plot: average distance over time
  df_dist = pd.concat([rng_to_par_to_results[r][par]['time_dist']
                       for r in main_rng_seeds], axis=1)
  df_dist = df_dist.fillna(method='ffill').mean(axis=1)
  ax[0].plot(df_dist, lw=1, label=label)
  # Second plot: MAPE over iterations
  df_mape = pd.concat([rng_to_par_to_results[r][par]['avg_mape']
                       for r in main_rng_seeds], axis=1).fillna(method='ffill')
  df_mape = df_mape.fillna(method='ffill').mean(axis=1)
  ax[1].plot(df_mape, marker='o', label=label)

# Other plot goodies
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
# Save global plot
plot_file = os.path.join(root_output_folder,
                         f'par_vs_{indep_seq_runs}_all.png')
fig.savefig(plot_file, bbox_inches='tight', dpi=300)

# Compute scalar global metrics
print("Global metrics:")
for par in parallelism_levels:
  nums_unfeas = [ rng_to_par_to_results[r][par]['avg_n_unfeas']
                  for r in main_rng_seeds ]
  avg_dists = [ rng_to_par_to_results[r][par]['avg_dist']
                for r in main_rng_seeds ]
  avg_dists_fea_unf = [ rng_to_par_to_results[r][par]['avg_dist_fea_unf']
                        for r in main_rng_seeds ]
  print(f"par = {par}: avg_n_unfeas = {np.mean(nums_unfeas)}, "
        f"avg_dist = {np.mean(avg_dists)}, "
        f"avg_dist_fea_unf = {np.mean(avg_dists_fea_unf)}")
