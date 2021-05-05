import sys
from deep_rl import *
import subprocess
from importlib import reload


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


# DOE
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
  kwargs['game'] = config.game
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
  run_steps(DoeAgent(config))


# sa integer option single action policy
def sa_single_net(**kwargs):
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
  kwargs['game'] = config.game
  generate_tag(kwargs)
  config.merge(kwargs)

  SingleSANet = reload(sys.modules['deep_rl.network.network_heads']).SingleSANet
  config.network_fn = lambda: SingleSANet(
      config.state_dim,
      config.action_dim,
      num_options=config.num_o,
      config=config)
  DoeAgent = reload(sys.modules['deep_rl.agent.DOE_agent']).DoeAgent
  run_steps(DoeAgent(config))


if __name__ == "__main__":
  random_seed()
  set_one_thread()
  select_device(-1)
  cf = Config()
  # cf.merge()

  cf.params_set = 'benchmark'
  cf.run = 4410
  # DOE
  kwargs = dict(run=cf.run, params_set=cf.params_set)
  # doe_continuous(**kwargs)
  sa_single_net(**kwargs)
