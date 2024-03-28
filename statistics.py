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
parallelism_levels = [10, 1]
indep_seq_runs = 10
num_runs = 5
root_rng_seed = 20230524
opt_constraints = {'RMSD_0.75': (0, 2.1)}
target_col = '-RMSD^3*TIME'
root_output_folder = os.path.join('outputs',
                                  'simulated_p10_init5')
df_all_file = os.path.join('resources', 'ligen', 'ligen_synth_table.csv')
regret_ylim_single = 2500
regret_ylim_avg = 2500

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
    # SCALAR METRICS
    ## Percentage of unfeasible executions including initial points
    perc_unfeas_dic = dict.fromkeys(group_seeds, None)
    ## Number of feasible executions including initial points
    n_feas_dic = dict.fromkeys(group_seeds, None)
    ## Number of unfeasible executions excluding initial points
    n_unfeas_noinit_dic = dict.fromkeys(group_seeds, None)
    ## Average regret on feasible points over all iterations
    avg_regr_dic = dict.fromkeys(group_seeds, None)
    ## Average regret on all points (both feasible and unfeasible) over all
    ## iterations
    avg_regr_fea_unf_dic = dict.fromkeys(group_seeds, None)
    ## Sums of execution times of target functions (unfeasible and total)
    exec_times_unfeas_dic = dict.fromkeys(group_seeds, None)
    exec_times_total_dic = dict.fromkeys(group_seeds, None)
    # VECTORS OF METRICS
    ## MAPE on all iterations
    mape_dic = dict.fromkeys(group_seeds, None)
    ## Regret over time (one value per second)
    time_regr_dic = dict.fromkeys(group_seeds, None)
    ## Regret over iterations
    iters_regr_dic = dict.fromkeys(group_seeds, None)
    ## dict of time elapsed at end of an execution -> feasibility of execution
    time_feas_dic = {}

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

      # Collect *feasible* incumbents at each iteration
      points = []
      curr = np.nan
      for i in range(hist.shape[0]):
        if feas.iloc[i] and (curr is np.nan
                             or hist['target'].iloc[i] < curr):
          curr = hist['target'].iloc[i]
        points.append(curr)
      regrets = pd.Series(points, index=hist.index)

      # Regret vector considering both feasible & unfeasible points
      regr_fea_unf = hist['target']

      # Get optimizer times for each evaluation
      sorted_regrets = regrets.loc[0:].reset_index(drop=True)
      sorted_regrets.name = 'regret'
      discrete_times = info['optimizer_time']
      iters_regr_dic[rng] = pd.concat((discrete_times, sorted_regrets), axis=1)
      # Initialize vector of time instants
      delta = 1.0  # granularity
      time_grid = np.arange(0, discrete_times.iloc[-1]+delta, delta)
      time_regr = pd.Series(index=time_grid)
      # Collect current regret at each time instant in the grid
      for i in range(hist.index.max()):
        time_regr[discrete_times[i]:discrete_times[i+1]] = regrets[i]
      time_regr[discrete_times.iloc[-1]:] = regrets.iloc[-1]

      # Collect time -> feasibility
      for t, feas_t in zip(discrete_times, feas):
        time_feas_dic[t] = feas_t

      # Execution times of target functions (recall: f(x) = RMSD^3(x) * T(x))
      exec_times = hist['target'] / hist['RMSD_0.75'] ** 3
      exec_times_unfeas_dic[rng] = exec_times[~feas].sum()
      exec_times_total_dic[rng] = exec_times.sum()

      # Add stuff to results dictionaries
      perc_unfeas_dic[rng] = (~feas).sum() / hist.shape[0]
      n_feas_dic[rng] = feas.sum()
      n_unfeas_noinit_dic[rng] = ( (~feas.loc[feas.index != -1]).sum() )
      avg_regr_dic[rng] = regrets.mean()
      avg_regr_fea_unf_dic[rng] = regr_fea_unf.cummin().mean()
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
    group_time_regr_worst = time_regr_df.max(axis=1).loc[:end_time]
    # Compute regret over iterations for the group
    iters_regr = pd.concat(iters_regr_dic.values(),
                             axis=0, ignore_index=True)
    iters_regr = iters_regr.sort_values(by='optimizer_time') \
                           .reset_index(drop=True)['regret'].cummin()

    # Collect scalar metrics for the group
    par_to_results[par]['avg_perc_unfeas'] = \
                                      np.mean(list(perc_unfeas_dic.values()))
    par_to_results[par]['avg_n_unfeas_noinit'] = \
                       np.mean(list(n_unfeas_noinit_dic.values()))
    par_to_results[par]['avg_regr'] = np.mean(list(avg_regr_dic.values()))
    par_to_results[par]['avg_regr_fea_unf'] = \
                       np.mean(list(avg_regr_fea_unf_dic.values()))
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
    par_to_results[par]['time_regr'] = group_time_regr
    par_to_results[par]['time_regr_all'] = time_regr_dic
    par_to_results[par]['time_regr_worst'] = group_time_regr_worst
    par_to_results[par]['time_feas'] = time_feas_dic
    par_to_results[par]['iters_regr'] = iters_regr
    par_to_results[par]['end_time'] = end_time
    # Write all group metrics into the global results dict
    rng_to_par_to_results[main_rng] = par_to_results

  # We have looped on all parallelism levels.
  # Now, for the current main RNG seed, we print and plot stuff
  fig, ax = plt.subplots(3, 1, figsize=(8, 12))
  for par in parallelism_levels:
    label = f"parallelism {par}"
    par_n_unf = par_to_results[par]['avg_perc_unfeas']
    par_n_unf_noinit = par_to_results[par]['avg_n_unfeas_noinit']
    exec_times_unfeas = par_to_results[par]['exec_times_unfeas']
    exec_times_total = par_to_results[par]['exec_times_total']
    exec_time_over_nfeas = par_to_results[par]['exec_time_over_nfeas']

    # Regret and feasibility of executions over time
    times_regrs = par_to_results[par]['time_regr']
    ax[0].plot(times_regrs, lw=1, label=label)
    color = ax[0].lines[-1].get_color()
    # Plot individual agents in the case of parallelism 1
    if par == 1:
      e_t = par_to_results[par]['end_time']
      for t_d in par_to_results[par]['time_regr_all'].values():
        ax[0].plot(t_d.loc[:e_t], lw=0.25, color=color)

    # # Regret over iterations
    # ax[1].plot(par_to_results[par]['iters_regr'], label=label)

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
      ax[2].set_title("Ranking of centralized model vs ensembles")
      ax[2].set_xlabel("time [s]")
      ax[2].set_ylabel("ranking")
      ax[2].grid(axis='y', alpha=0.4)

  # Other plot goodies
  ax[0].axhline(ground, c='lightgreen', ls='--', label="ground truth",
                        zorder=-2)
  ax[0].set_xlabel("time [s]")
  ax[0].grid(axis='y', alpha=0.4)
  ax[0].set_ylim(None, regret_ylim_single)
  ax[0].set_title("Target values of incumbents")
  ax[0].legend()
  # ax[1].axhline(ground, c='lightgreen', ls='--', label="ground truth",
  #                       zorder=-2)
  # ax[1].set_xlabel("iterations")
  # ax[1].grid(axis='y', alpha=0.4)
  # ax[1].set_ylim(None, regret_ylim_single)
  # ax[1].set_title("Target values of incumbents")
  # ax[1].legend()
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
fig, ax = plt.subplots(3, 1, figsize=(8, 12))
for par in parallelism_levels:
  num_indep_runs = rng_to_par_to_results[main_rng_seeds[0]] \
                                        [par]['num_indep_runs']
  label = f"parallelism {par}"
  # Average regret over time
  df_time_regr = pd.concat([rng_to_par_to_results[r][par]['time_regr']
                            for r in main_rng_seeds], axis=1)
  df_time_regr = df_time_regr.fillna(method='ffill').mean(axis=1)
  df_regr_worst = pd.concat([rng_to_par_to_results[r][par]['time_regr_worst']
                             for r in main_rng_seeds], axis=1)
  df_regr_worst = df_regr_worst.fillna(method='ffill').mean(axis=1)
  ax[0].plot(df_time_regr, lw=1.5, label=label)
  color = ax[0].lines[-1].get_color()
  ## Plot all ensembles or only ensemble bounds
  if par == 1:
    label_ensemble = f"{label} (ensemble max.)"
    ax[0].plot(df_regr_worst, lw=0.25, label=label_ensemble, color=color)
  elif par == indep_seq_runs:
    df_par_async = df_time_regr.copy()

  # # Regret over iterations
  # df_iters_regr = pd.concat([rng_to_par_to_results[r][par]['iters_regr']
  #                            for r in main_rng_seeds], axis=1).mean(axis=1)
  # ax[1].plot(df_iters_regr, lw=1.5, label=label)

  # MAPE over iterations (only if available)
  try:
    df_mape = pd.concat([rng_to_par_to_results[r][par]['avg_mape']
                     for r in main_rng_seeds], axis=1).fillna(method='ffill')
    df_mape = df_mape.fillna(method='ffill').mean(axis=1)
    ax[1].plot(df_mape, label=label)
  except:
    pass

# Make rankings plot
avg_ranking = pd.concat(par_async_rankings.values(), axis=1).mean(axis=1)
ax[2].plot(avg_ranking)
ax[2].set_title("Ranking of centralized model vs ensembles")
ax[2].set_xlabel("time [s]")
ax[2].set_ylabel("ranking")
ax[2].grid(axis='y', alpha=0.4)

# Other plot goodies
ax[0].axhline(ground, c='lightgreen', ls='--', label="ground truth", zorder=-2)
ax[0].set_xlabel("time [s]")
ax[0].set_ylim(None, regret_ylim_avg)
ax[0].grid(axis='y', alpha=0.4)
ax[0].set_title("Target values of incumbents")
ax[0].legend()
# ax[1].axhline(ground, c='lightgreen', ls='--', label="ground truth", zorder=-2)
# ax[1].set_xlabel("iterations")
# ax[1].grid(axis='y', alpha=0.4)
# ax[1].set_ylim(None, regret_ylim_single)
# ax[1].set_title("Target values of incumbents")
# ax[1].legend()
ax[1].set_xlabel("iterations")
ax[1].grid(axis='y', alpha=0.4)
ax[1].set_title("Training MAPE")
ax[1].legend()
fig.subplots_adjust(hspace=0.25)
# Save global plot
suffix = os.path.basename(root_output_folder)
plot_file = os.path.join(root_output_folder, f'00_{suffix}.png')
fig.savefig(plot_file, bbox_inches='tight', dpi=300)
exit()

# Compute scalar global metrics, print them and save them to file
strg = "Global metrics:\n"
for par in parallelism_levels:
  nums_unfeas = [ rng_to_par_to_results[r][par]['avg_perc_unfeas']
                  for r in main_rng_seeds ]
  nums_unfeas_noinit = [ rng_to_par_to_results[r][par]['avg_n_unfeas_noinit']
                  for r in main_rng_seeds ]
  avg_regrs = [ rng_to_par_to_results[r][par]['avg_regr']
                for r in main_rng_seeds ]
  avg_regrs_fea_unf = [ rng_to_par_to_results[r][par]['avg_regr_fea_unf']
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
           f"avg_regr = {np.mean(avg_regrs)}, "
           f"avg_regr_fea_unf = {np.mean(avg_regrs_fea_unf)}, "
           f"exec_time_over_nfeas = {np.mean(exec_times_over_nf)}, "
           f"exec_times_unfeas = "
           f"{np.mean(exec_times_ratio).round(3)}\n")
print(strg)
res_file = os.path.join(root_output_folder, 'results.txt')
with open(res_file, 'w') as f:
  f.write(strg)
