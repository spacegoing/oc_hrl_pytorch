from multiprocessing import Pool
import subprocess
import time
import sys
import traceback
import os
from deep_rl import *
from dev_dwsa import wsa
from importlib import reload
from pymongo import MongoClient
client = MongoClient('mongodb://localhost:27017')
db = client['sa']
error_col = db['wsa_jobs_error']


def call_job(run_game):
  print('Start task: ', run_game)
  run, game, idx = run_game
  try:
    kwargs = dict(run=run, params_set=game)
    wsa(**kwargs)
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
  num_proc = 24
  num_run = 12
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
  # games = ['benchmarklog']
  # games = ['antlog', 'humanoidstanduplog']
  # games = ['t_walker2']
  # games = ['dmwalker2_s']  # 6081; 6000
  # games = ['dmwalker2_l']  # 6091
  # games = [
  #     'dmwalker2', 'dmtcartpole', 'dmtreacher', 'dmtfish', 'dmtcheetah',
  #     'dmtwalker1'
  # ]
  # # num_worker=1 dm=20 6500
  # # num_worker=4 dm=20 6504; dm=30 6534
  # # num_worker=4 dm=20 4024 fix embed but not P_o
  # # cartpole 20>+30; reacher 30; cheetah 20 > +30; fish 20; walker1 20+30<; walker2 30
  # games = [
  #     'dm-hopper', 'dm-acrobot', 'dm-finger', 'dm-humanoid-w', 'dm-humanoid-r',
  #     'dm-manipulator', 'dm-quadruped', 'dm-stacker', 'dm-swimmer'
  # ]
  # nlayer = 1
  # run 400: qfn linear lag=10
  # run 410: qfn FFN lag=10
  # run 403: qfn linear lag=3
  # run 413: qfn FFN lag=3
  # nlayer = 3
  # run 4303: qfn linear lag=3
  # run 4300: qfn linear lag=10
  # run 4313: qfn FFN lag=3
  # run 4310: qfn FFN lag=10

  # nlayer = 1
  # 1 layer, 3 skill lag, vfn 2 linear
  # run 51323: vfn = linear(linear)
  # 1 layer, 5 skill lag, vfn 2 linear
  # run 51523: vfn = linear(linear)

  # run 8890 gae_tau=0.90
  # run 8885 gae_tau=0.85
  # run 8880 gae_tau=0.80
  # run 8875 gae_tau=0.75
  # run 8870 gae_tau=0.70
  games = [
      'benchmark',
      'humanoidstandup',
      # 'walker',
      # 'humanoid4',
      # 'ant',
  ]
  games = [
      'benchmark',
      'swimmer4',
      'reacher',
      'humanoidstandup',
  ]
  # run 1111: same settings with upload branch
  # run 1120: act_mean_lc act_std_lc with ot_tilde_a detached
  # run 1121: act_mean_lc act_std_lc
  # run 1400: WsaFFN use encoder's output rather than [ot,st] concat
  # run 1401: with ot_tilde_a detached
  # run 1501: same with 1401 but
  #             P(ot|st,ot_1,k) ot_1,k is not detached

  # run 1003: skill_lag = 3
  # run 1005: skill_lag = 5
  # run 1001: skill_lag = 1
  run_list = [[1001, game, i] for game in games for i in range(num_run)]
  with Pool(processes=num_proc) as pool:
    start = time.time()
    for x in pool.imap(call_job, run_list):
      print("(Time elapsed: {}s)".format(int(time.time() - start)))
