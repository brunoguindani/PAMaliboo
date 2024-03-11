import numpy as np
import opentuner
from opentuner import ConfigurationManipulator, EnumParameter, \
                      IntegerParameter, MeasurementInterface, Result
from opentuner.measurement.inputmanager import FixedInputManager
from opentuner.search.objective import ThresholdAccuracyMinimizeTime
import os
import pandas as pd
import time


# Function to find approximated configuration in the given dataset
def find_approximation_idx(dataset, x, num_features):
  min_distance = None
  approximations = []
  approximations_idxs = []

  # dataset_configs only contains configurations, one per row
  dataset_configs = dataset.iloc[:, 0:num_features].values
  # Loop over dataset rows
  for idx in range(dataset_configs.shape[0]):
    # Compute distance between row and current best value
    row = dataset_configs[idx, 0:num_features]
    dist = np.linalg.norm(x - row, 2)
    if min_distance is None or dist <= min_distance:
      if dist == min_distance:
        # One of the tied best approximations
        approximations.append(row)
        approximations_idxs.append(dataset.index[idx])
      else:
        # The one new best approximation
        min_distance = dist
        approximations = [row]
        approximations_idxs = [dataset.index[idx]]

  # If multiple, choose randomly
  internal_idx = np.random.randint(0, len(approximations_idxs))
  x_idx = approximations_idxs[internal_idx]
  # x_approx = approximations[internal_idx]
  return x_idx


# Base class for constrained optimization
class ConstrainedTuner(MeasurementInterface):
  is_constrained = True
  file = ''
  num_features = 0
  minimizing_column = ''
  constrained_column = ''
  constraints = []
  iterations = 0 + 0
  """
  if accuracy >= target:
    minimize time
  else:
    maximize accuracy
  """
  def __init__(self, args):
    inp_man = FixedInputManager()
    objective = ThresholdAccuracyMinimizeTime(threshold_lower_bound)
    super().__init__(args, input_manager=inp_man, objective=objective)

  def manipulator(self):
    """Define search space"""
    raise NotImplementedError()

  def run(self, desired_result, input, limit):
    """Run given configuration and return performance"""
    dataset = pd.read_csv(self.file)
    x = np.array([d for d in desired_result.configuration.data.values()])
    x_idx = find_approximation_idx(dataset, x, self.num_features)
    row_approx = dataset.loc[x_idx]
    minimizing_val = -int(row_approx[self.minimizing_column])
    constrained_val = int(row_approx[self.constrained_column])
    x_strg = ','.join(row_approx.astype(str))
    time.sleep(row_approx['TIME_TOTAL']/10)
    with open(output_file, 'a') as f:
      f.write(f"{x_idx},{x_strg}\n")
    with open(info_file, 'a') as f:
      f.write(f"{x_idx},{time.time()-time_start}\n")
    return Result(time=minimizing_val, accuracy=-constrained_val)


class LigenTuner(ConstrainedTuner):
  file = os.path.join('resources', 'ligen', 'ligen_synth_table.csv')
  num_features = 8
  minimizing_column = '-RMSD^3*TIME'
  constrained_column = 'RMSD_0.75'
  constraints = [2.1]
  iterations = 10 + 1000

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


if __name__ == '__main__':
  # Choose experiment class
  experiment_tuner = LigenTuner

  # Command line arguments
  argparser = opentuner.default_argparser()
  namespace = argparser.parse_args()
  namespace.parallelism = 10
  namespace.test_limit = experiment_tuner.iterations

  # Initialize relevant variables
  #experiment_name = experiment_tuner.file.replace('.csv', '')
  list_feat = ['index'] \
              + [f'x{_}' for _ in range(experiment_tuner.num_features)]
  list_feat.extend(['RMSD_0.75', '-RMSD^3*TIME'] \
                   if experiment_tuner.is_constrained else ['target'])
  output_folder = 'outputs_opentuner'
  os.makedirs(output_folder)

  # Loop over RNG seeds and constraint thresholds
  root_rng_seed = 20230524
  for rng in range(root_rng_seed, root_rng_seed+10):
    threshold_lower_bound = experiment_tuner.constraints[0]
    # Initialize results file
    output_rng_folder = os.path.join(output_folder, f'rng{rng}')
    os.makedirs(output_rng_folder)
    output_file = os.path.join(output_rng_folder, 'history.csv')
    info_file = os.path.join(output_rng_folder, 'info.csv')
    with open(output_file, 'w') as f:
      f.write(','.join(list_feat) + '\n')
    with open(info_file, 'w') as f:
      f.write('index,optimizer_time\n')
    # Run optimizer
    time_start = time.time()
    experiment_tuner.main(namespace)
