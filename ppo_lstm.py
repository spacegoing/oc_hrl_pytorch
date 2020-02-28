#######################################################################
# Copyright (C) 2017 Shangtong Zhang(zhangshangtong.cpp@gmail.com)    #
# Permission given to modify the code as long as you keep this        #
# declaration at the top                                              #
#######################################################################
from deep_rl import *

LeastSaveInterval = 50000


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


def ppo_continuous(**kwargs):
  generate_tag(kwargs)
  kwargs.setdefault('log_level', 0)
  config = Config()
  config.merge(kwargs)

  config.num_workers = 9  # must greater than 3
  config.single_process = True
  config.task_fn = lambda: Task(
      config.game,
      num_envs=config.num_workers,
      single_process=config.single_process)
  config.eval_env = Task(config.game)

  config.network_fn = lambda: GaussianActorCriticNet(
      config.state_dim,
      config.action_dim,
      actor_body=FCBody(config.state_dim, gate=torch.tanh),
      critic_body=FCBody(config.state_dim, gate=torch.tanh))
  config.optimizer_fn = lambda params: torch.optim.Adam(params, 3e-4, eps=1e-5)
  config.discount = 0.99
  config.use_gae = True
  config.gae_tau = 0.95
  config.gradient_clip = 0.5
  config.rollout_length = 2048
  config.optimization_epochs = 10
  config.mini_batch_size = 64
  config.ppo_ratio_clip = 0.2
  config.log_interval = 2048
  config.state_normalizer = MeanStdNormalizer()
  run_steps(PPOAgent(config))


# PPOC
def ppoc_continuous(**kwargs):
  kwargs['algo_tag'] = 'PPOC'
  generate_tag(kwargs)
  kwargs.setdefault('log_level', 0)
  kwargs.setdefault('num_o', 4)
  kwargs.setdefault('gate', nn.ReLU())
  kwargs.setdefault('entropy_weight', 0.01)
  kwargs.setdefault('tasks', False)
  kwargs.setdefault('max_steps', 2e6)
  config = Config()
  config.merge(kwargs)

  if config.tasks:
    set_tasks(config)

  if 'dm-humanoid' in config.game:
    hidden_units = (128, 128)
  else:
    hidden_units = (64, 64)

  config.num_workers = 9  # must greater than 3
  config.single_process = True
  config.task_fn = lambda: Task(
      config.game,
      num_envs=config.num_workers,
      single_process=config.single_process)
  config.eval_env = Task(config.game)

  config.network_fn = lambda: OptionGaussianActorCriticNet(
      config.state_dim,
      config.action_dim,
      num_options=config.num_o,
      actor_body=FCBody(
          config.state_dim, hidden_units=hidden_units, gate=config.gate),
      critic_body=FCBody(
          config.state_dim, hidden_units=hidden_units, gate=config.gate),
      option_body_fn=lambda: FCBody(
          config.state_dim, hidden_units=hidden_units, gate=config.gate),
  )
  config.optimizer_fn = lambda params: torch.optim.Adam(params, 3e-4, eps=1e-5)
  config.beta_reg = 0.01
  config.discount = 0.99
  config.use_gae = True
  config.gae_tau = 0.95
  config.gradient_clip = 0.5
  config.rollout_length = 30
  config.optimization_epochs = 10
  config.mini_batch_size = 64
  config.ppo_ratio_clip = 0.2
  config.log_interval = 2048
  config.state_normalizer = MeanStdNormalizer()
  run_steps(PPOCLSTMAgent(config))


def ppoc_lstm_continuous(**kwargs):
  config = Config()
  config.use_gae = False
  config.rollout_length = 256
  gstr = 'False'
  if config.use_gae:
    gstr = 'True'
  kwargs['algo_tag'] = 'PPOC_AbE_LSTM_GAE_%s_%d' % (gstr, config.rollout_length)
  generate_tag(kwargs)
  kwargs.setdefault('log_level', 0)
  kwargs.setdefault('learning', 'all')
  kwargs.setdefault('tasks', False)
  config.merge(kwargs)

  config.num_workers = 9
  config.single_process = True
  config.task_fn = lambda: Task(
      config.game,
      num_envs=config.num_workers,
      single_process=config.single_process)
  config.eval_env = Task(config.game)

  config.num_o = 4
  if 'dm-humanoid' in config.game:
    hidden_units = (128, 128)
  else:
    hidden_units = (64, 64)
  config.gate = nn.ReLU()

  # lstm parameters
  config.debug = False
  config.hid_dim = 64
  config.num_lstm_layers = 1
  config.lstm_to_fc_feat_dim = config.num_lstm_layers * config.hid_dim
  config.bi_direction = True
  if config.bi_direction:
    config.lstm_to_fc_feat_dim = config.lstm_to_fc_feat_dim * 2
  config.lstm_dropout = 0
  config.network_fn = lambda: OptionLstmGaussianActorCriticNet(
      config.state_dim,
      config.action_dim,
      num_options=config.num_o,
      hid_dim=config.hid_dim,
      phi_body=DummyBody(config.state_dim),
      option_body_fn=lambda: FCBody(
          config.state_dim, hidden_units=hidden_units, gate=config.gate),
      config=config)
  config.optimizer_fn = lambda params: torch.optim.Adam(params, 3e-4, eps=1e-5)
  config.gradient_clip = 0.5

  config.discount = 0.99
  config.state_normalizer = MeanStdNormalizer()
  config.log_interval = 2048
  config.save_interval = config.rollout_length * config.num_workers * 10
  while config.save_interval <= LeastSaveInterval:
    config.save_interval += config.save_interval

  # PPO params
  # training params
  config.optimization_epochs = 10
  config.mini_batch_size = 64
  # model params
  config.gae_tau = 0.95
  config.ppo_ratio_clip = 0.2
  config.entropy_weight = 0.01

  # OC params
  config.beta_reg = 0.01
  config.delib_cost = 0.01

  run_steps(PPOCLSTMAgent(config))


def ppo_lstm_continuous(**kwargs):
  config = Config()
  config.use_gae = True
  config.rollout_length = 32 * 2
  gstr = 'False'
  if config.use_gae:
    gstr = 'True'
  kwargs['algo_tag'] = 'PPO_AbE_LSTM_GAE_%s_%d' % (gstr, config.rollout_length)
  generate_tag(kwargs)
  kwargs.setdefault('log_level', 0)
  config.merge(kwargs)

  config.num_workers = 9  # must greater than 3
  config.single_process = True
  config.task_fn = lambda: Task(
      config.game,
      num_envs=config.num_workers,
      single_process=config.single_process)
  config.eval_env = Task(config.game)

  # lstm parameters
  config.hid_dim = 64
  config.num_lstm_layers = 1
  config.lstm_to_fc_feat_dim = config.num_lstm_layers * config.hid_dim
  config.bi_direction = True
  if config.bi_direction:
    config.lstm_to_fc_feat_dim = config.lstm_to_fc_feat_dim * 2
  config.lstm_dropout = 0
  config.network_fn = lambda: LstmActorCriticNet(
      config.state_dim,
      config.action_dim,
      config.hid_dim,
      actor_body=FCBody(config.lstm_to_fc_feat_dim, gate=torch.tanh),
      critic_body=FCBody(config.lstm_to_fc_feat_dim, gate=torch.tanh),
      config=config)

  config.optimizer_fn = lambda params: torch.optim.Adam(params, 3e-4, eps=1e-5)
  config.gradient_clip = 0.5

  config.discount = 0.99
  config.state_normalizer = MeanStdNormalizer()
  config.log_interval = 2048
  config.save_interval = config.rollout_length * config.num_workers * 10
  while config.save_interval <= LeastSaveInterval:
    config.save_interval += config.save_interval

  # PPO params
  # training params
  config.optimization_epochs = 10
  config.mini_batch_size = 64
  # model params
  config.gae_tau = 0.95
  config.ppo_ratio_clip = 0.2
  config.entropy_weight = 0.01

  run_steps(PPOAgent(config))


# if __name__ == '__main__':
if True:
  # use tuna to profile:
  # python -m cProfile -o program.prof run_ppoc.py
  mkdir('log/oc')
  mkdir('tf_log/oc')
  mkdir('data')
  set_one_thread()
  random_seed()
  # select_device(-1)
  select_device(1)
  env_list = [
      'RoboschoolHopper-v1', 'RoboschoolWalker2d-v1',
      'RoboschoolHalfCheetah-v1', 'RoboschoolAnt-v1', 'RoboschoolHumanoid-v1'
  ]

  game = 'CartPole-v0'
  # dqn_feature(game=game)
  # quantile_regression_dqn_feature(game=game)
  # categorical_dqn_feature(game=game)
  # a2c_feature(game=game)
  # n_step_dqn_feature(game=game)
  # option_critic_feature(game=game)
  # ppo_feature(game=game)

  # game = 'HalfCheetah-v2'
  game = 'RoboschoolHopper-v1'
  # game = 'BipedalWalkerHardcore-v2'
  game = 'LunarLanderContinuous-v2'
  # oc_continuous(game=game)
  # doc_continuous(game=game)
  # a2c_continuous(game=game)
  # ppo_lstm_continuous(game=game)
  # ppo_continuous(game=game)
  # ddpg_continuous(game=game)
  # td3_continuous(game=game)
  game = 'CSI300-v1'
  ppo_lstm_continuous(game=game)
  ppoc_lstm_continuous(game=game)
  ppoc_continuous(game=game)

  game = 'BreakoutNoFrameskip-v4'
  # dqn_pixel(game=game)
  # quantile_regression_dqn_pixel(game=game)
  # categorical_dqn_pixel(game=game)
  # a2c_pixel(game=game)
  # n_step_dqn_pixel(game=game)
  # option_critic_pixel(game=game)
  # ppo_pixel(game=game)
