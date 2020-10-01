# -*- coding: utf-8 -*-
import sys
from deep_rl import *
import subprocess
from importlib import reload
import numpy as np


def get_closest_fname(fname, step, config):
  fname = fname[:-14]
  # list for 2048 rollout
  step_list = np.array([
      i * config.save_interval
      for i in range(int(config.max_steps // config.save_interval) + 1)
  ])
  print('All step list:', step_list)
  diff_step = np.abs(step_list - step)
  closest_idx = diff_step.argmin()
  closest_step = step_list[closest_idx]
  print('Closest to input %d: %d' % (step, closest_step))
  fname = 'DoeAgent-' + fname + '-' + str(closest_step)
  return fname, closest_step


def set_tasks(config):
  if config.game == 'dm-walker':
    tasks = ['walk', 'run']
  elif config.game == 'dm-walker-1':
    tasks = ['squat', 'stand']
    config.game = 'dm-walker'
  elif config.game == 'dm-walker-2':
    tasks = ['walk', 'backward']
    config.game = 'dm-walker'
  elif config.game == 'dm-finger':
    tasks = ['turn_easy', 'turn_hard']
  elif config.game == 'dm-reacher':
    tasks = ['easy', 'hard']
  elif config.game == 'dm-cartpole-b':
    tasks = ['balance', 'balance_sparse']
    config.game = 'dm-cartpole'
  elif config.game == 'dm-cartpole-s':
    tasks = ['swingup', 'swingup_sparse']
    config.game = 'dm-cartpole'
  elif config.game == 'dm-fish':
    tasks = ['upright', 'downleft']
  elif config.game == 'dm-hopper':
    tasks = ['stand', 'hop']
  elif config.game == 'dm-acrobot':
    tasks = ['swingup', 'swingup_sparse']
  elif config.game == 'dm-manipulator':
    tasks = ['bring_ball', 'bring_peg']
  elif config.game == 'dm-cheetah':
    tasks = ['run', 'backward']
  else:
    raise NotImplementedError

  games = ['%s-%s' % (config.game, t) for t in tasks]
  config.tasks = [Task(g, num_envs=config.num_workers) for g in games]
  config.game = games[0]


def doe_continuous(**kwargs):
  config = basic_doe_params()

  config.merge(kwargs)
  config.merge(doe_params_dict.get(kwargs.get('params_set'), dict()))

  if config.tasks:
    set_tasks(config)

  config.task_fn = lambda: Task(config.game, num_envs=config.num_workers)
  config.eval_env = Task(config.game)

  if 'dm-humanoid' in config.game:
    config.nhid = 128

  kwargs['remark'] = 'Param_%s_Net_nhead%d_dm%d_nl%d_nhid%d_nO_%d' %\
    (kwargs.get('params_set',''),
     config.nhead, config.dmodel, config.nlayers, config.nhid,config.num_o)
  generate_tag(kwargs)
  config.merge(kwargs)

  DoeContiOneOptionNet = reload(
      sys.modules['deep_rl.network.network_heads']).DoeContiOneOptionNet
  config.network_fn = lambda: DoeContiOneOptionNet(
      config.state_dim,
      config.action_dim,
      num_options=config.num_o,
      nhead=config.nhead,
      dmodel=config.dmodel,
      nlayers=config.nlayers,
      nhid=config.nhid,
      dropout=0.2,
      config=config)
  DoeAgent = reload(sys.modules['deep_rl.agent.DOE_agent']).DoeAgent
  agent = DoeAgent(config)

  # inputs
  fname = 'HalfCheetah-v2-params_set_save_model_debug-remark_Param_save_model_debug_Net_nhead4_dm40_nl3_nhid50-tasks_False-run-1-200826-221958'
  fname = 'DoeAgent-HalfCheetah-v2-params_set_benchmarklog-remark_Param_benchmarklog_Net_nhead1_dm40_nl1_nhid50_nO_4-run-4660-491520'
  step = 983040
  data_dir = './data/'
  env = config.task_fn()

  # load model
  fname, closest_step = get_closest_fname(fname, step, config)
  fname = 'DoeAgent-HalfCheetah-v2-params_set_benchmarklog-remark_Param_benchmarklog_Net_nhead1_dm40_nl1_nhid50_nO_4-run-4660-983040'
  agent.load(data_dir + fname)

  # rollout episode
  out_dir = data_dir + '%s_episode_%d/' % (config.game, closest_step)

  from pymongo import MongoClient
  client = MongoClient('mongodb://localhost:27017')
  db = client['sa']
  nm = 'HalfCheetah-v2-params_set_benchmarklog-remark_Param_benchmarklog_Net_nhead1_dm40_nl1_nhid50_nO_4-run-4660-200927-095741-94'
  col = db[nm]
  state_list = list(
      col.find({}, {
          "_id": 0,
          "s": 1,
          "step": 1
      }).sort([("step", -1)]).limit(1))
  states = tensor(np.array(state_list[0]['s']))

  sample_mean_list = []
  for i in range(20):
    with torch.no_grad():
      # ot_hat: [num_workers]
      ot_hat = tensor(np.zeros([4]))
      ot_hat[:] = 3
      ot_hat = ot_hat.long()
      ## beginning of actions part
      # ot: v_t [num_workers, dmodel(embedding size in init)]
      ot = agent.network.embed_option(ot_hat.unsqueeze(0)).detach().squeeze(0)
      ot[:, 0] += -0.1 + 0.01 * i
      # obs_cat: [num_workers, state_dim + dmodel]
      obs = states
      obs_cat = torch.cat([obs, ot], dim=-1)
      # obs_hat: \tilde{S}_t [1, num_workers, dmodel]
      # obs_hat = agent.act_concat_norm(obs_cat)
      obs_hat = obs_cat
      # generate batch inputs for each option
      pat_mean, pat_std = agent.network.act_doe(obs_hat)
      sample_mean_list.append(to_np(pat_mean))
  sample_mean_mat = np.array(sample_mean_list)
  mat = sample_mean_mat[:, 0, :].T

  with open('sampled_action.npy', 'wb') as f:
    np.save(f, mat)

  param_gen = agent.network.named_parameters()
  for n, p in param_gen:
    print(n)
    print(p)
    break

  subprocess.run(
      ['ffmpeg', '-i',
       '%s/%%04d.png' % (out_dir),
       '%s.gif' % (out_dir)])
  # with open('%s_options.bin' % (sub_folder), 'wb') as f:
  #   pickle.dump(agent.all_options, f)


if __name__ == "__main__":
  random_seed()
  set_one_thread()
  select_device(-1)
  cf = Config()
  cf.params_set = 'benchmarklog'

  # DOE
  kwargs = dict(
      game='HalfCheetah-v2', run=cf.run, params_set=cf.params_set, nhead=4)
  doe_continuous(**kwargs)
