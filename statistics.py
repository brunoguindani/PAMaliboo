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

from matplotlib.lines import Line2D
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd


# Campaign parameters
use_relative = False
use_incumbents = True
plot_all_ensembles = False
parallelism_levels = [10, 1]
indep_seq_runs = 10
num_runs = 10
root_rng_seed = 20230524
opt_constraints = {'RMSD_0.75': (0, 2.1)}
target_col = '-RMSD^3*TIME'
root_output_folder = os.path.join('outputs', 'synth_SVR_p10_init5')
df_all_file = os.path.join('resources', 'ligen', 'ligen_synth_table.csv')
if 'SVR' in root_output_folder:
  mape_ylim = 0.2
elif 'error' in root_output_folder:
  mape_ylim = 0.5
else:
  mape_ylim = 0.1

# Find real feasible optimum
df_all = pd.read_csv(df_all_file)
## NOTE: this script changes the signs of the target function, and looks for
## the minimum value. This is different than the library code, which seeks to
## maximize the target value
df_all[target_col] *= -1
## Find feasible points
df_all['feas'] = True
for key, (lb, ub) in opt_constraints.items():
  df_all['feas'] = df_all['feas'] & (lb <= df_all[key]) & (df_all[key] <= ub)
df_feas = df_all[ df_all['feas'] == True ]
## Find point with minimum target value among those points
best = df_feas.loc[ df_feas[target_col] == df_feas[target_col].min() ].iloc[0]
ground = 0 if use_relative else best[target_col]

# Initialize main RNG seeds
main_rng_seeds = [root_rng_seed+i for i in range(num_runs)]
rng_to_par_to_results = {m: {} for m in main_rng_seeds}

for main_rng in main_rng_seeds:
  # Initialize dict of global results for this main RNG seed
  par_to_results = {p: {} for p in parallelism_levels}
  for par in parallelism_levels:
    # For independent sequential experiments, each main RNG seed has a *group*
    # of indep_seq_runs linked RNG seeds, which includes the main seed itself.
    # Otherwise, the group reduces to just the main seed
    group_seeds = [main_rng]
    if par == 1 and indep_seq_runs > 1:
      other_seeds = [10*main_rng+i for i in range(indep_seq_runs-1)]
      group_seeds.extend(other_seeds)
    print(f">>> par = {par}, main_rng = {main_rng}, -> {group_seeds}")

    # Initialize results dictionaries for this group: each entry links an RNG
    # seed in the group (i.e. an individual run) to a certain measurement.
    # By 'distance' we mean either relative regret or raw target values,
    # according to the value of use_relative
    # SCALAR METRICS
    ## Percentage of unfeasible executions including initial points
    perc_unfeas_dic = dict.fromkeys(group_seeds, None)
    ## Number of feasible executions including initial points
    n_feas_dic = dict.fromkeys(group_seeds, None)
    ## Number of unfeasible executions excluding initial points
    n_unfeas_noinit_dic = dict.fromkeys(group_seeds, None)
    ## Average distance on feasible points over all iterations
    avg_dist_dic = dict.fromkeys(group_seeds, None)
    ## Average distance on all points (both feasible and unfeasible) over all
    ## iterations
    avg_dist_fea_unf_dic = dict.fromkeys(group_seeds, None)
    ## Sums of execution times of target functions (unfeasible and total)
    exec_times_unfeas_dic = dict.fromkeys(group_seeds, None)
    exec_times_total_dic = dict.fromkeys(group_seeds, None)
    # VECTORS OF METRICS
    ## MAPE on all iterations
    mape_dic = dict.fromkeys(group_seeds, None)
    ## Distance over time
    time_dist_dic = dict.fromkeys(group_seeds, None)
    ## Time elapsed at end of an execution -> feasibility of the execution
    time_feas_dic = {}
    ## Elapsed time after computing first configuration via BO (i.e. duration
    ## of initial points exploration + some acquisition overhead)
    initial_times_dic = dict.fromkeys(group_seeds, None)

    # Loop over individual RNG seeds in this group
    for rng in group_seeds:
      # Get history dataframe for this experiment
      output_folder = os.path.join(root_output_folder, f'par_{par}',
                                                       f'rng_{rng}')
      hist = pd.read_csv(os.path.join(output_folder, 'history.csv'),
                         index_col='index')
      ## We change the signs here as well, since we want to minimize
      hist['target'] *= -1
      # Also get the recorded additional information
      info = pd.read_csv(os.path.join(output_folder, 'info.csv'),
                         index_col='index')
      if 'opentuner' in output_folder:  # TODO meh...
        hist.reset_index(drop=True, inplace=True)
        info.reset_index(drop=True, inplace=True)

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
                               or hist['target'].iloc[i] < curr):
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
        # ...simple relative regret wrt the true minimum
        dists = (points - best[target_col]) / best[target_col]
      else:
        # ...or target value (which we are minimizing)
        dists = points

      # Distance vector considering both feasible & unfeasible points, either:
      if use_relative:
        # ...simple relative regret wrt the true minimum
        dist_fea_unf = (hist['target'] - best[target_col]) / best[target_col]
      else:
        # ...or target value (which we are minimizing)
        dist_fea_unf = hist['target']

      # Get optimizer times for each evaluation
      discrete_times = info['optimizer_time']
      # Initialize vector of time instants
      delta = 1.0  # granularity
      time_grid = np.arange(0, discrete_times.iloc[-1]+delta, delta)
      time_dist = pd.Series(index=time_grid)
      # Collect current distance at each time instant in the grid
      for i in range(hist.index.max()):
        time_dist[discrete_times[i]:discrete_times[i+1]] = dists[i]
      time_dist[discrete_times.iloc[-1]:] = dists.iloc[-1]

      # Collect time -> feasibility
      for t, feas_t in zip(discrete_times, feas):
        time_feas_dic[t] = feas_t

      # # Shrink optimizer time based on parallelism level
      # if isinstance(par, int):
      #   time_dist.index = time_dist.index / par
      #   time_feas_dic = {t/par: f for t, f in time_feas_dic.items()}

      # Execution times of target functions (recall: f(x) = RMSD^3(x) * T(x))
      exec_times = hist['target'] / hist['RMSD_0.75'] ** 3
      exec_times_unfeas_dic[rng] = exec_times[~feas].sum()
      exec_times_total_dic[rng] = exec_times.sum()

      # Add stuff to results dictionaries
      perc_unfeas_dic[rng] = (~feas).sum() / hist.shape[0]
      n_feas_dic[rng] = feas.sum()
      n_unfeas_noinit_dic[rng] = ( (~feas.loc[feas.index != -1]).sum() )
      avg_dist_dic[rng] = dists.mean()
      avg_dist_fea_unf_dic[rng] = dist_fea_unf.cummin().mean()
      if 'train_MAPE' in info.columns:
        mape_dic[rng] = info['train_MAPE']
      time_dist_dic[rng] = time_dist
      initial_times_dic[rng] = info.loc[0, 'optimizer_time']

    # We have looped on RNG seeds of this group.
    # Now we combine vectors of metrics into single DataFrames
    group_avg_mape = pd.concat(mape_dic.values(), axis=1).mean(axis=1) \
                     if 'train_MAPE' in info.columns else None
    ## In group_time_dist, we take the best (smallest) value of the group at
    ## each time instant
    group_time_dist = pd.concat(time_dist_dic.values(), axis=1).min(axis=1)
    # Stop at the earliest time at which a seed in the group has finished
    end_time = np.min([d.index[-1] for d in time_dist_dic.values()])
    group_time_dist = group_time_dist.loc[:end_time]

    # Collect scalar metrics for the group
    par_to_results[par]['avg_perc_unfeas'] = \
                                      np.mean(list(perc_unfeas_dic.values()))
    par_to_results[par]['avg_n_unfeas_noinit'] = \
                       np.mean(list(n_unfeas_noinit_dic.values()))
    par_to_results[par]['avg_dist'] = np.mean(list(avg_dist_dic.values()))
    par_to_results[par]['avg_dist_fea_unf'] = \
                       np.mean(list(avg_dist_fea_unf_dic.values()))
    par_to_results[par]['num_indep_runs'] = len(group_seeds)
    par_to_results[par]['indep_runs_iters'] = hist.shape[0]
    exec_times_unfeas = sum(exec_times_unfeas_dic.values())
    exec_times_total = sum(exec_times_total_dic.values())
    n_feas = sum(n_feas_dic.values())
    par_to_results[par]['exec_times_unfeas'] = exec_times_unfeas
    par_to_results[par]['exec_times_total'] = exec_times_total
    par_to_results[par]['exec_time_over_nfeas'] = exec_times_total / n_feas
    # Collect combined vectors of metrics for the group
    par_to_results[par]['avg_mape'] = group_avg_mape
    par_to_results[par]['time_dist'] = group_time_dist
    par_to_results[par]['time_dist_all'] = time_dist_dic
    par_to_results[par]['time_feas'] = time_feas_dic
    par_to_results[par]['end_time'] = end_time
    # Initial exploration times
    par_to_results[par]['initial_times'] = initial_times_dic
    # Write all group metrics into the global results dict
    rng_to_par_to_results[main_rng] = par_to_results

  # We have looped on all parallelism levels.
  # Now, for the current main RNG seed, we print and plot stuff
  print(f"For main RNG seed {main_rng}:")
  fig, ax = plt.subplots(2, 1, figsize=(8, 8))
  for par in parallelism_levels:
    par_n_unf = par_to_results[par]['avg_perc_unfeas']
    par_n_unf_noinit = par_to_results[par]['avg_n_unfeas_noinit']
    exec_times_unfeas = par_to_results[par]['exec_times_unfeas']
    exec_times_total = par_to_results[par]['exec_times_total']
    exec_time_over_nfeas = par_to_results[par]['exec_time_over_nfeas']

    label = f"parallelism {par}"
    print(f"par = {par}: avg_perc_unfeas = {round(par_n_unf, 3)}, "
          f"avg_n_unfeas_noinit = {par_n_unf_noinit}, "
          f"avg_dist = {round(par_to_results[par]['avg_dist'], 3)}, "
          f"exec_time_over_nfeas = {exec_time_over_nfeas}, "
          f"exec_times_unfeas = "
          f"{round(exec_times_unfeas / exec_times_total, 3)}")

    # Comment from here for not creating the plots
    # First plot: distance and feasibility of executions over time
    times_dists = par_to_results[par]['time_dist']
    ax[0].plot(times_dists, lw=1,
                            label=label+f" (unfeasible: {par_n_unf:.3f})")
    color = ax[0].lines[-1].get_color()
    # Plot individual agents in the case of parallelism 1
    if par == 1:
      for t_d in par_to_results[par]['time_dist_all'].values():
        e_t = par_to_results[par]['end_time']
        ax[0].plot(t_d.loc[:e_t], lw=0.5, color=color)
    # Loop on iterations (their optimizer time and feasibility)
    for t, feas in par_to_results[par]['time_feas'].items():
      # Compute approximation of optimizer time on the time grid
      if t <= times_dists.index[-1]:
        time_approx = times_dists.index[times_dists.index > t][0]
      else:
        # if beyond last grid element, approximate to last grid element
        time_approx = times_dists.index[-1]
      # Plot iteration differently according to its feasibility
      facecolor = color if feas else 'none'
      ax[0].scatter(time_approx, times_dists[time_approx], marker='o', s=12,
                    edgecolor=color, facecolors=facecolor, linewidths=0.5)
    # Draw lines for initial exploration times
    for t_init in par_to_results[par]['initial_times'].values():
      ax[0].axvline(t_init, ls='-.', lw=0.5, color=color)

    # Second plot: MAPE over iterations (only if available)
    if par_to_results[par]['avg_mape'] is not None:
      ax[1].plot(par_to_results[par]['avg_mape'], marker='o', label=label)
  # Other plot goodies
  ## For first plot
  ax[0].axhline(ground, c='lightgreen', ls='--', label="ground truth",
                        zorder=-2)
  title_part1 = "Relative regret" if use_relative else "Target values"
  title_part2 = "incumbents" if use_incumbents else "points"
  title_full = f"{title_part1} of {title_part2}"
  ax[0].set_xlabel("time [s]")
  ax[0].grid(axis='y', alpha=0.4)
  ax[0].set_title(title_full)
  ax[0].legend()
  handles, labels = ax[0].get_legend_handles_labels()
  handles.append(Line2D([0], [0], ls='-.', lw=0.5, color='gray'))
  labels.append("start of BO")
  ax[0].legend(handles=handles, labels=labels)
  # For second plot
  ax[1].set_xlabel("iterations")
  ax[1].set_ylim(-0.01, mape_ylim)
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
fig, ax = plt.subplots(3, 1, figsize=(8, 12))
for par in parallelism_levels:
  num_indep_runs = rng_to_par_to_results[main_rng_seeds[0]] \
                                        [par]['num_indep_runs']
  label = f"parallelism {par}"
  # First plot: average distance over time
  df_dist = pd.concat([rng_to_par_to_results[r][par]['time_dist']
                       for r in main_rng_seeds], axis=1)
  df_dist = df_dist.fillna(method='ffill').mean(axis=1)
  ax[0].plot(df_dist, lw=1.5, label=label)
  color = ax[0].lines[-1].get_color()
  avg_init_time = np.mean(list(par_to_results[par]['initial_times'].values()))
  ax[0].axvline(avg_init_time, ls='-.', lw=0.5, color=color)
  ## Plot all ensembles or only ensemble bounds
  if par == 1:
    df_all_ensembles = pd.concat([rng_to_par_to_results[r][par]['time_dist']
                                  for r in main_rng_seeds], axis=1) \
                         .fillna(method='ffill', axis=0)
    if plot_all_ensembles:
      label_ensemble = f"{label} (individual seed)"
      for r in main_rng_seeds:
        ax[0].plot(rng_to_par_to_results[r][par]['time_dist'], lw=0.25,
                   label=label_ensemble, color=color)
        label_ensemble = None
    else:
      label_ensemble = f"{label} (ensemble bounds)"
      ax[0].plot(df_all_ensembles.min(axis=1), lw=0.5, color=color,
                                                       label=label_ensemble)
      ax[0].plot(df_all_ensembles.max(axis=1), lw=0.5, color=color)
  elif par == indep_seq_runs:
    df_par_async = df_dist.copy()

  # Second plot: MAPE over iterations (only if available)
  try:
    df_mape = pd.concat([rng_to_par_to_results[r][par]['avg_mape']
                     for r in main_rng_seeds], axis=1).fillna(method='ffill')
    df_mape = df_mape.fillna(method='ffill').mean(axis=1)
    ax[1].plot(df_mape, marker='o', label=label)
  except:
    pass

# Make rankings plot
if len(parallelism_levels) > 1:
  df_par_async = df_par_async.loc[df_all_ensembles.index]
  # This counts how many ensembles beat the PA model at time t, for each t.
  # By adding 1, one obtains the ranking of the PA model wrt the ensembles
  comp = lambda x : x < df_par_async
  df_ranking = 1 + df_all_ensembles.apply(comp, axis=0).sum(axis=1)
  ax[2].plot(df_ranking)
  ax[2].set_title("Ranking of centralized model vs ensembles")
  ax[2].set_xlabel("time [s]")
  ax[2].set_ylabel("ranking")
  ax[2].set_yticks(np.arange(1, num_runs+1.01, 1))
  ax[2].grid(axis='y', alpha=0.4)

# Other plot goodies
## For first plot
ax[0].axhline(ground, c='lightgreen', ls='--', label="ground truth", zorder=-2)
ax[0].set_xlabel("time [s]")
ax[0].grid(axis='y', alpha=0.4)
ax[0].set_title(title_full)
ax[0].legend()
handles, labels = ax[0].get_legend_handles_labels()
handles.append(Line2D([0], [0], ls='-.', lw=0.5, color='gray'))
labels.append("start of BO")
ax[0].legend(handles=handles, labels=labels)
## For second plot
ax[1].set_xlabel("iterations")
ax[1].set_ylim(-0.01, mape_ylim)
ax[1].grid(axis='y', alpha=0.4)
ax[1].set_title("Training MAPE")
ax[1].legend()
fig.subplots_adjust(hspace=0.25)
# Save global plot
try:
  suffix = os.path.basename(root_output_folder).split('_init')[-1].zfill(3)
except:
  suffix = 'opentuner' if 'opentuner' in output_folder else 'results'
plot_file = os.path.join(root_output_folder, f'all_{suffix}.png')
fig.savefig(plot_file, bbox_inches='tight', dpi=300)
# Comment until here for not creating the plots

# Compute scalar global metrics, print them and save them to file
strg = "Global metrics:\n"
for par in parallelism_levels:
  nums_unfeas = [ rng_to_par_to_results[r][par]['avg_perc_unfeas']
                  for r in main_rng_seeds ]
  nums_unfeas_noinit = [ rng_to_par_to_results[r][par]['avg_n_unfeas_noinit']
                  for r in main_rng_seeds ]
  avg_dists = [ rng_to_par_to_results[r][par]['avg_dist']
                for r in main_rng_seeds ]
  avg_dists_fea_unf = [ rng_to_par_to_results[r][par]['avg_dist_fea_unf']
                        for r in main_rng_seeds ]
  exec_times_unfeas = [ rng_to_par_to_results[r][par]['exec_times_unfeas']
                        for r in main_rng_seeds ]
  exec_times_total = [ rng_to_par_to_results[r][par]['exec_times_total']
                       for r in main_rng_seeds ]
  exec_times_ratio = [u/t for u, t in zip(exec_times_unfeas, exec_times_total)]
  exec_times_over_nf = [ rng_to_par_to_results[r][par]['exec_time_over_nfeas']
                          for r in main_rng_seeds ]
  strg += (f"par = {par}: avg_perc_unfeas = {np.mean(nums_unfeas)}, "
           f"avg_n_unfeas_noinit = {np.mean(nums_unfeas_noinit)}, "
           f"avg_dist = {np.mean(avg_dists)}, "
           f"avg_dist_fea_unf = {np.mean(avg_dists_fea_unf)}, "
           f"exec_time_over_nfeas = {np.mean(exec_times_over_nf)}, "
           f"exec_times_unfeas = "
           f"{np.mean(exec_times_ratio).round(3)}\n")
print(strg)
res_file = os.path.join(root_output_folder, 'results.txt')
with open(res_file, 'w') as f:
  f.write(strg)
