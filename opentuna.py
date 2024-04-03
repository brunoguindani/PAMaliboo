import numpy as np
import opentuner
from opentuner import ConfigurationManipulator, EnumParameter, \
                      IntegerParameter, MeasurementInterface, Result
from opentuner.measurement.inputmanager import FixedInputManager
from opentuner.search.objective import ThresholdAccuracyMinimizeTime
import os
import pandas as pd
import time


class LigenTuner(MeasurementInterface):
  """
  if accuracy >= target:
    minimize time
  else:
    maximize accuracy
  """
  file = os.path.join('resources', 'ligen', 'ligen_synth_table.csv')
  num_features = 8
  minimizing_col = '-RMSD^3*TIME'
  constrained_col = 'RMSD_0.75'
  exec_time_col = 'TIME_TOTAL'
  constraints = [2100]  # 2.1 multiplied by 1000 since OT only handles integers
  iterations = 5 + 1000

  def __init__(self, args):
    inp_man = FixedInputManager()
    objective = ThresholdAccuracyMinimizeTime(threshold_lower_bound)
    self.df = pd.read_csv(self.file)
    self.domain = self.df.iloc[:, 0:self.num_features].to_numpy()
    self.history = pd.DataFrame(columns=
                    'ALIGN_SPLIT,OPTIMIZE_SPLIT,OPTIMIZE_REPS,'
                    'CUDA_THREADS,N_RESTART,CLIPPING,SIM_THRESH,BUFFER_SIZE,'
                    'target,RMSD_0.75,evaluation_time'.split(','))
    self.info = pd.DataFrame(columns=['optimizer_time'])
    self.parallelism = args.parallelism
    self.queue = []
    self.the_clock = 0
    super().__init__(args, input_manager=inp_man, objective=objective)

  def manipulator(self):
    """Define search space"""
    manipulator = ConfigurationManipulator()
    manipulator.add_parameter(EnumParameter('ALIGN_SPLIT',
                                            [8, 12, 16, 20, 24, 32, 48, 72]))
    manipulator.add_parameter(EnumParameter('OPTIMIZE_SPLIT',
                                            [8, 12, 16, 20, 24, 32, 48, 72]))
    manipulator.add_parameter(IntegerParameter('OPTIMIZE_REPS', 1, 5))
    manipulator.add_parameter(EnumParameter('CUDA_THREADS',
                              [32, 64, 96, 128, 160, 192, 224, 256]))
    manipulator.add_parameter(EnumParameter('N_RESTART', [256, 1024]))
    manipulator.add_parameter(EnumParameter('CLIPPING', [10, 30, 50, 256]))
    manipulator.add_parameter(IntegerParameter('SIM_THRESH', 1, 4))
    manipulator.add_parameter(EnumParameter('BUFFER_SIZE',
                [1048576, 2097152, 5242880, 10485760, 20971520, 52428800]))
    return manipulator

  def run(self, desired_result, input, limit):
    """Run given configuration and return performance"""
    # Find best approximation
    x = np.array([d for d in desired_result.configuration.data.values()])
    distances = np.linalg.norm(self.domain - x, axis=1)
    dataset_idx = np.argmin(distances)
    # Build string from dataset row
    x_approx = self.domain[dataset_idx,:]
    row_approx = self.df.loc[dataset_idx]
    x_all = np.hstack((x_approx,
                       row_approx[[self.minimizing_col,
                                   self.constrained_col, self.exec_time_col]]))
    # Avdance clock and write other stuff to the information file
    evaluation_time = row_approx[self.exec_time_col]
    if len(self.queue) < self.parallelism:
      self.queue.append(evaluation_time)
    else:
      idx_queue_min = np.argmin(self.queue)
      time_delta = self.queue[idx_queue_min]
      self.queue = [(q-time_delta) for q in self.queue]
      del self.queue[idx_queue_min]
      self.the_clock += time_delta
    print(self.queue)
    history_idx = time.time()
    self.history.loc[history_idx] = x_all
    self.info.loc[history_idx] = [self.the_clock]
    # Compute integer values for optimizer
    minimizing_val = -int(row_approx[self.minimizing_col])
    constrained_val = int(1000*row_approx[self.constrained_col])
    return Result(time=minimizing_val, accuracy=-constrained_val)

  def save_final_config(self, config):
    self.the_clock = 0
    self.history.sort_index(inplace=True)
    self.history.reset_index(drop=True, inplace=True)
    self.history.to_csv(history_file, index_label='index')
    self.info.sort_index(inplace=True)
    self.info.reset_index(drop=True, inplace=True)
    self.info.to_csv(info_file, index_label='index')


if __name__ == '__main__':
  # Command line arguments
  argparser = opentuner.default_argparser()
  namespace = argparser.parse_args()
  namespace.parallelism = 10
  namespace.test_limit = LigenTuner.iterations

  # Initialize relevant variables
  output_folder = os.path.join('outputs', 'opentuner_simulated')

  # Loop over RNG seeds and constraint thresholds
  root_rng_seed = 20230524
  for rng in range(root_rng_seed, root_rng_seed+5):
    threshold_lower_bound = LigenTuner.constraints[0]
    # Initialize results file
    output_rng_folder = os.path.join(output_folder, 'par_10', f'rng_{rng}')
    os.makedirs(output_rng_folder)
    history_file = os.path.join(output_rng_folder, 'history.csv')
    info_file = os.path.join(output_rng_folder, 'info.csv')
    # Run optimizer
    np.random.seed(rng)
    LigenTuner.main(namespace)
