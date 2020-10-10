from multiprocessing import Pool
import subprocess
import time
import sys
from deep_rl import *
from importlib import reload
from dev_doe import doe_continuous
import traceback
from pymongo import MongoClient
client = MongoClient('mongodb://localhost:27017')
db = client['sa']
error_col = db['mp_jobs_error']


def call_job(run_game):
  print('Start task: ', run_game)
  run, game, idx = run_game
  try:
    kwargs = dict(run=run, params_set=game)
    doe_continuous(**kwargs)
  except Exception as e:
    error_col.insert_one({
        'games_settings': run_game,
        'error': str(e),
        'tradeback': str(traceback.format_exc())
    })


if __name__ == "__main__":
  random_seed()
  set_one_thread()
  select_device(-1)
  # run_walker_list = [[4410, 'walkerlog', i] for i in range(12)]
  # run_walker_list += [[4410, 'benchmarklog', i] for i in range(12)]
  # with Pool(processes=17) as pool:
  #   start = time.time()
  #   for x in pool.imap(call_job, run_walker_list):
  #     print("(Time elapsed: {}s)".format(int(time.time() - start)))

  # run_swimmer_list = [[4410, 'swimmer', i] for i in range(12)]
  # run_swimmer_list += [[4410, 'hopper', i] for i in range(12)]
  # with Pool(processes=17) as pool:
  #   start = time.time()
  #   for x in pool.imap(call_job, run_swimmer_list):
  #     print("(Time elapsed: {}s)".format(int(time.time() - start)))
  games = [
      'ant',
      'humanoid4',
      'inverteddoublependulum',
      'invertedpendulum4',
      'walker',
      'hopper',
      'walker4',
      'hopper4',
      # 'benchmark',
      # 'reacher',
      # 'humanoidstandup',
      # 'humanoidstandup4',
      # 'reacher4',
  ]
  run_id = 4440
  # run_list = [[run_id, 'benchmarklog', i] for i in range(12)]
  run_list = [[run_id, game, i] for i in range(12) for game in games]
  # run_list += [[run_id, game, i] for i in range(12) for game in games]
  with Pool(processes=18) as pool:
    start = time.time()
    for x in pool.imap(call_job, run_list):
      print("(Time elapsed: {}s)".format(int(time.time() - start)))
