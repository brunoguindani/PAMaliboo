import os
import sys
import time

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pamaliboo.jobs import HyperqueueJobSubmitter


def test_jobs():
  submitter = HyperqueueJobSubmitter('output_test')
  id1 = submitter.submit(['./dummy.sh', '3', '4'], 'output_test/1.stdout')
  print("Submitted", id1)
  stat1 = submitter.get_job_status(id1)
  print("Status:", stat1)

  id2 = submitter.submit(['./dummy.sh', '5', '1'], 'output_test/2.stdout')
  print("Submitted", id2)

  print("Sleeping...")
  time.sleep(3)

  stat1 = submitter.get_job_status(id1)
  stat2 = submitter.get_job_status(id2)
  print("Statuses:", id1, stat1, "-", id2, stat2)
