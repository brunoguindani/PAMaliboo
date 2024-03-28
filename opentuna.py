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
    time_start = time.time()
    # Find best approximation
    x = np.array([d for d in desired_result.configuration.data.values()])
    distances = np.linalg.norm(self.domain - x, axis=1)
    idx = np.argmin(distances)
    # Build string from dataset row
    x_approx = self.domain[idx,:]
    row_approx = self.df.loc[idx]
    x_all = np.hstack((idx, x_approx,
                       row_approx[[self.minimizing_col,
                                   self.constrained_col, self.exec_time_col]]))
    x_strg = ','.join([str(_) for _ in x_all])
    # Write configuration to the history
    with open(history_file, 'a') as f:
      f.write(x_strg + '\n')
    # Avdance clock and write other stuff to the information file
    processing_time = time.time() - time_start
    evaluation_time = row_approx[self.exec_time_col]
    self.the_clock += processing_time + evaluation_time
    with open(info_file, 'a') as f:
      f.write(f'{idx},{self.the_clock},{processing_time}\n')
    # Compute integer values for optimizer
    minimizing_val = -int(row_approx[self.minimizing_col])
    constrained_val = int(1000*row_approx[self.constrained_col])
    return Result(time=minimizing_val, accuracy=-constrained_val)


if __name__ == '__main__':
  # Command line arguments
  argparser = opentuner.default_argparser()
  namespace = argparser.parse_args()
  namespace.parallelism = 10
  namespace.test_limit = LigenTuner.iterations

  # Initialize relevant variables
  history_header = ('index,ALIGN_SPLIT,OPTIMIZE_SPLIT,OPTIMIZE_REPS,'
                    'CUDA_THREADS,N_RESTART,CLIPPING,SIM_THRESH,BUFFER_SIZE,'
                    'target,RMSD_0.75,evaluation_time')
  info_header = 'index,optimizer_time,processing_time'
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
    with open(history_file, 'w') as f:
      f.write(history_header + '\n')
    with open(info_file, 'w') as f:
      f.write(info_header + '\n')
    # Run optimizer
    LigenTuner.main(namespace)
    LigenTuner.the_clock = 0
