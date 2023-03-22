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

from abc import ABC, abstractmethod
from enum import Enum
import json
import os
import subprocess
import warnings


class JobStatus(Enum):
  OTHER = 0
  CANCELED = 1
  FAILED = 2
  FINISHED = 3
  RUNNING = 4
  WAITING = 5


class JobSubmitter(ABC):
  def __init__(self, output_folder: str):
    self.output_folder = output_folder
    os.makedirs(self.output_folder, exist_ok=True)

  @abstractmethod
  def submit(self, cmd: list[str], output_file: str) -> int:
    pass

  @abstractmethod
  def get_job_status(self, job_id: int) -> JobStatus:
    pass



class HyperqueueJobSubmitter(JobSubmitter):
  hq_exec = 'lib/hq'

  def submit(self, cmd: list[str], output_file: str) -> int:
    """Submit a job using Hyperqueue"""
    file_path = os.path.join(self.output_folder, output_file)
    hq_cmd = [self.hq_exec, 'submit', '--output-mode', 'json',
              '--stdout', file_path, '--stderr', '/dev/null'] + cmd
    sub = subprocess.run(hq_cmd, capture_output=True)
    # TODO handle errors in json loading and 'id' access
    output = json.loads(sub.stdout.decode())
    return output['id']


  def get_job_status(self, job_id: int) -> JobStatus:
    cmd = [self.hq_exec, 'job', 'list', '--all', '--output-mode', 'json']
    sub = subprocess.run(cmd, capture_output=True)
    # TODO handle errors in json loading and [] accesses
    output = json.loads(sub.stdout.decode())
    for job in output:
      if job['id'] == job_id:
        job_stats = job['task_stats']
        if job_stats['canceled'] == 1:
          return JobStatus.CANCELED
        elif job_stats['failed'] == 1:
          return JobStatus.FAILED
        elif job_stats['finished'] == 1:
          return JobStatus.FINISHED
        elif job_stats['running'] == 1:
          return JobStatus.RUNNING
        elif job_stats['waiting'] == 1:
          return JobStatus.WAITING
        else:
          return JobStatus.OTHER

    raise RuntimeError(f"Job id {job_id} not found")
