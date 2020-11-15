from multiprocessing import Pool
import subprocess
import time
import sys
import traceback
import os
from deep_rl import *
from importlib import reload
from dev_doe import doe_continuous
from run_dac import dac_ppo
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


def dac_call_job(run_game):
  print('Start task: ', run_game)
  run, game, idx = run_game
  try:
    kwargs = dict(run=run, params_set=game)
    dac_ppo(**kwargs)
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
  num_proc = 5
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
      'benchmark',
      'swimmer',
      'humanoidstandup',
      'reacher',
      'walker',
      'hopper',
      'inverteddoublependulum',
      'invertedpendulum4',
      'humanoid4',
      'ant',
  ]
  games = ['benchmarklog']
  games = ['antlog', 'humanoidstanduplog']
  games = ['t_walker2']
  run_list = [[60001, game, i] for i in range(num_proc) for game in games]
  with Pool(processes=num_proc) as pool:
    start = time.time()
    for x in pool.imap(call_job, run_list):
    # for x in pool.imap(dac_call_job, run_list):
      print("(Time elapsed: {}s)".format(int(time.time() - start)))
