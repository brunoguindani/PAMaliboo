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
import sys


# Campaign parameters
parallelism_levels = [10, 1, 'opentuner']
indep_seq_runs = 10
num_runs = int(sys.argv[2])
root_rng_seed = 20230524
opt_constraints = {'RMSD_0.75': (0, 2.1)}
target_col = '-RMSD^3*TIME'
root_output_folder = os.path.join('outputs', sys.argv[1])
table_prefix = 'full' if 'full' in sys.argv[1] else 'synth'
df_all_file = os.path.join('resources', 'ligen',
                          f'ligen_{table_prefix}_table.csv')
regret_ylim_single = 2500
regret_ylim_avg = 2500
stop_time = 42000
time_delta = 1.0  # granularity of regret computation in time

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
ground = best[target_col]

# Initialize main RNG seeds and other containers
main_rng_seeds = [root_rng_seed+i for i in range(num_runs)]
rng_to_par_to_results = {m: {} for m in main_rng_seeds}
par_async_rankings = dict.fromkeys(main_rng_seeds, None)
par_to_labels = {1: 'EMaliboo', 10: 'PAMaliboo', 'opentuner': 'OpenTuner'}
print("Saving plots to", root_output_folder)

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
    print(f">>> par = {par}, main_rng = {main_rng} -> {group_seeds}")

    # Initialize results dictionaries for this group: each entry links an RNG
    # seed in the group (i.e. an individual run) to a certain measurement
    ## MAPE on all iterations
    mape_dic = dict.fromkeys(group_seeds, None)
    ## Regret over time (one value per second)
    time_regr_dic = dict.fromkeys(group_seeds, None)

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
      if 'opentuner' in output_folder:
        hist.reset_index(drop=True, inplace=True)
        info.reset_index(drop=True, inplace=True)

      # Check feasibility of observations with respect to the constraints
      feas = pd.Series(True, index=hist.index)
      for key, (lb, ub) in opt_constraints.items():
        feas = feas & (lb <= hist[key]) & (hist[key] <= ub)

      # Collect *feasible* incumbents at each iteration
      points = []
      curr = np.nan
      for i in range(hist.shape[0]):
        if feas.iloc[i] and (curr is np.nan
                             or hist['target'].iloc[i] < curr):
          curr = hist['target'].iloc[i]
        points.append(curr)
      regrets = pd.Series(points, index=hist.index)

      # Get optimizer times for each evaluation
      discrete_times = info['optimizer_time']
      # Initialize vector of time instants
      time_grid = np.arange(0, discrete_times.iloc[-1]+time_delta, time_delta)
      time_regr = pd.Series(index=time_grid)
      # Collect current regret at each time instant in the grid
      for i in range(hist.index.max()):
        time_regr[discrete_times[i]:discrete_times[i+1]] = regrets[i]
      time_regr[discrete_times.iloc[-1]:] = regrets.iloc[-1]

      # Add stuff to results dictionaries
      if 'train_MAPE' in info.columns:
        mape_dic[rng] = info['train_MAPE']
      time_regr_dic[rng] = time_regr

    # We have looped on RNG seeds of this group.
    # Now we combine vectors of metrics into single DataFrames
    group_avg_mape = pd.concat(mape_dic.values(), axis=1).mean(axis=1) \
                     if 'train_MAPE' in info.columns else None
    ## In group_time_regr, we take the best (smallest) value of the group at
    ## each time instant
    time_regr_df = pd.concat(time_regr_dic.values(), axis=1)
    # Stop at the earliest time at which a seed in the group has finished
    end_time = np.min([d.index[-1] for d in time_regr_dic.values()])
    group_time_regr = time_regr_df.min(axis=1).loc[:end_time]

    # Collect scalar metrics for the group
    par_to_results[par]['num_indep_runs'] = len(group_seeds)
    # Collect combined vectors of metrics for the group
    par_to_results[par]['avg_mape'] = group_avg_mape
    par_to_results[par]['time_regr'] = group_time_regr
    par_to_results[par]['time_regr_all'] = time_regr_dic
    par_to_results[par]['end_time'] = end_time
    # Write all group metrics into the global results dict
    rng_to_par_to_results[main_rng] = par_to_results

  # We have looped on all parallelism levels.
  # Now, for the current main RNG seed, we print and plot stuff
  fig, ax = plt.subplots(3, 1, figsize=(8, 12))
  for par in parallelism_levels:
    label = par_to_labels[par]

    # Regret and feasibility of executions over time
    times_regrs = par_to_results[par]['time_regr']
    if stop_time > times_regrs.index[-1]:
      # extend to `stop_time` if it has not been reached (only in this plot)
      new_times = np.arange(0, stop_time+time_delta, time_delta)
      times_regrs = times_regrs.reindex(new_times, method='ffill')
    ax[0].plot(times_regrs, lw=1, label=label)
    ax[0].set_xlim(0, stop_time)
    color = ax[0].lines[-1].get_color()
    # Plot individual agents in the case of parallelism 1
    if par == 1:
      e_t = par_to_results[par]['end_time']
      for t_d in par_to_results[par]['time_regr_all'].values():
        ax[0].plot(t_d.loc[:e_t], lw=0.25, color=color)

    # MAPE over iterations (only if available)
    if par_to_results[par]['avg_mape'] is not None:
      ax[1].plot(par_to_results[par]['avg_mape'], label=label)

    # Rankings
    # This counts how many ensemble agents beat the PA model at time t, for
    # each t. By adding 1 we obtain the ranking of the PA model wrt the agents
    if par == indep_seq_runs:
      end_time = min(par_to_results[1]['end_time'],
                     par_to_results[indep_seq_runs]['end_time'])
      df_p1 = pd.concat(par_to_results[1]['time_regr_all'].values(), axis=1) \
                .loc[:end_time]
      df_pisr = par_to_results[indep_seq_runs]['time_regr_all'][main_rng] \
                .loc[:end_time]
      comp = lambda x : x < df_pisr
      df_ranking = 1 + df_p1.apply(comp, axis=0).sum(axis=1)
      par_async_rankings[main_rng] = df_ranking
      ax[2].plot(df_ranking)
      ax[2].set_title("Ranking of centralized model vs ensemble members")
      ax[2].set_xlabel("time [s]")
      ax[2].set_ylabel("ranking")
      ax[2].set_xlim(0, stop_time)
      ax[2].grid(axis='y', alpha=0.4)

  # Other plot goodies
  ax[0].axhline(ground, c='lightgreen', ls='--', label="ground truth",
                        zorder=-2)
  ax[0].set_xlabel("time [s]")
  ax[0].grid(axis='y', alpha=0.4)
  ax[0].set_ylim(None, regret_ylim_single)
  ax[0].set_title("Target values of incumbents")
  ax[0].legend()
  ax[1].set_xlabel("iterations")
  ax[1].grid(axis='y', alpha=0.4)
  ax[1].set_title("Training MAPE")
  ax[1].legend()
  fig.subplots_adjust(hspace=0.25)
  # Save plot to file
  plot_file = os.path.join(root_output_folder, f'{main_rng}.png')
  fig.savefig(plot_file, bbox_inches='tight', dpi=300)
  print()

# Make global plot
figsize = (8, 4)
fig_a, ax_a = plt.subplots(figsize=figsize)
fig_b, ax_b = plt.subplots(figsize=figsize)
fig_c, ax_c = plt.subplots(figsize=figsize)
for par in parallelism_levels:
  num_indep_runs = rng_to_par_to_results[main_rng_seeds[0]] \
                                        [par]['num_indep_runs']
  label = par_to_labels[par]
  # Average regret over time
  df_time_regr = pd.concat([rng_to_par_to_results[r][par]['time_regr']
                            for r in main_rng_seeds], axis=1)
  df_time_regr = df_time_regr.fillna(method='ffill').mean(axis=1)
  stop_value = df_time_regr[stop_time]
  print(f"Incumbent for par = {par} at time {stop_time}: {stop_value}")
  ax_a.plot(df_time_regr, lw=1.5, label=label)

  # MAPE over iterations (only if available)
  try:
    df_mape = pd.concat([rng_to_par_to_results[r][par]['avg_mape']
                     for r in main_rng_seeds], axis=1).fillna(method='ffill')
    df_mape = df_mape.fillna(method='ffill').mean(axis=1)
    ax_b.plot(df_mape, label=label)
  except:
    pass

# Make rankings plot
avg_ranking = pd.concat(par_async_rankings.values(), axis=1).mean(axis=1)
ax_c.plot(avg_ranking)
ax_c.set_xlabel("time [s]")
ax_c.set_ylabel("ranking")
ax_c.set_xlim(0, stop_time)
ax_c.grid(axis='y', alpha=0.4)

# Other plot goodies
ax_a.axhline(ground, c='lightgreen', ls='--', label="ground truth", zorder=-2)
ax_a.set_xlabel("time [s]")
ax_a.set_xlim(0, stop_time)
ax_a.set_ylim(None, regret_ylim_avg)
ax_a.grid(axis='y', alpha=0.4)
ax_a.legend()
ax_b.set_xscale('log')
ax_b.set_xlabel("log(iterations)")
ax_b.grid(axis='y', alpha=0.4)
ax_b.legend()

# Save global plots
letters_to_figs = {'a': fig_a, 'b': fig_b, 'c': fig_c}
for letter, fig in letters_to_figs.items():
  basename = os.path.basename(root_output_folder)
  plot_file = os.path.join(root_output_folder, f'00_{basename}_{letter}.png')
  # fig.subplots_adjust(hspace=0.25)
  fig.savefig(plot_file, bbox_inches='tight', dpi=300)
