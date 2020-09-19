from multiprocessing import Pool
import subprocess
import time
import sys
from deep_rl import *
from importlib import reload
from dev_doe import doe_continuous


def call_job(run_game):
  print('Start task: ', run_game)
  run, game, idx = run_game
  try:
    kwargs = dict(run=run, params_set=game)
    doe_continuous(**kwargs)
  except:
    print('Error: ', run_game)


if __name__ == "__main__":
  random_seed()
  set_one_thread()
  select_device(-1)
  run_walker_list = [[4410, 'walkerlog', i] for i in range(12)]
  run_halfcheetah_list = [[4410, 'benchmarklog', i] for i in range(12)]
  with Pool(processes=3) as pool:
    start = time.time()
    for x in pool.imap(call_job, run_walker_list):
      print("(Time elapsed: {}s)".format(int(time.time() - start)))
    for x in pool.imap(call_job, run_halfcheetah_list):
      print("(Time elapsed: {}s)".format(int(time.time() - start)))

  run_swimmer_list = [[4410, 'swimmer', i] for i in range(12)]
  run_swimmer_list += [[4410, 'hopper', i] for i in range(12)]
  with Pool(processes=17) as pool:
    start = time.time()
    for x in pool.imap(call_job, run_swimmer_list):
      print("(Time elapsed: {}s)".format(int(time.time() - start)))
