#######################################################################
# Copyright (C) 2017 Shangtong Zhang(zhangshangtong.cpp@gmail.com)    #
# Permission given to modify the code as long as you keep this        #
# declaration at the top                                              #
#######################################################################
from deep_rl import *


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


# Option-Critic
def option_critic_feature(**kwargs):
  generate_tag(kwargs)
  kwargs.setdefault('log_level', 0)
  config = Config()
  config.merge(kwargs)

  config.num_workers = 5
  config.task_fn = lambda: Task(config.game, num_envs=config.num_workers)
  config.eval_env = Task(config.game)
  config.optimizer_fn = lambda params: torch.optim.RMSprop(params, 0.001)
  config.network_fn = lambda: OptionCriticNet(
      FCBody(config.state_dim), config.action_dim, num_options=2)
  config.random_option_prob = LinearSchedule(1.0, 0.1, 1e4)
  config.discount = 0.99
  config.target_network_update_freq = 200
  config.rollout_length = 5
  config.termination_regularizer = 0.01
  config.entropy_weight = 0.01
  config.gradient_clip = 5
  run_steps(OptionCriticAgent(config))


def option_critic_pixel(**kwargs):
  generate_tag(kwargs)
  kwargs.setdefault('log_level', 0)
  config = Config()
  config.merge(kwargs)

  config.task_fn = lambda: Task(config.game, num_envs=config.num_workers)
  config.eval_env = Task(config.game)
  config.num_workers = 16
  config.optimizer_fn = lambda params: torch.optim.RMSprop(
      params, lr=1e-4, alpha=0.99, eps=1e-5)
  config.network_fn = lambda: OptionCriticNet(
      NatureConvBody(), config.action_dim, num_options=4)
  config.random_option_prob = LinearSchedule(0.1)
  config.state_normalizer = ImageNormalizer()
  config.reward_normalizer = SignNormalizer()
  config.discount = 0.99
  config.target_network_update_freq = 10000
  config.rollout_length = 5
  config.gradient_clip = 5
  config.max_steps = int(2e7)
  config.entropy_weight = 0.01
  config.termination_regularizer = 0.01
  run_steps(OptionCriticAgent(config))


# Option-critic continuous
def oc_continuous(**kwargs):
  generate_tag(kwargs)
  kwargs.setdefault('log_level', 0)
  kwargs.setdefault('num_o', 4)
  kwargs.setdefault('learning', 'all')
  kwargs.setdefault('gate', nn.ReLU())
  kwargs.setdefault('entropy_weight', 0.01)
  kwargs.setdefault('tasks', False)
  kwargs.setdefault('max_steps', 2e6)
  kwargs.setdefault('num_workers', 16)
  config = Config()
  config.merge(kwargs)

  if 'dm-humanoid' in config.game:
    hidden_units = (128, 128)
  else:
    hidden_units = (64, 64)

  config.num_workers = 5
  config.task_fn = lambda: Task(config.game, num_envs=config.num_workers)
  config.eval_env = Task(config.game)
  config.optimizer_fn = lambda params: torch.optim.RMSprop(params, 0.001)
  config.network_fn = lambda: OptionGaussianActorCriticNet(
      config.state_dim,
      config.action_dim,
      num_options=config.num_o,
      phi_body=DummyBody(config.state_dim),
      actor_body=FCBody(
          config.state_dim, hidden_units=hidden_units, gate=config.gate),
      critic_body=FCBody(
          config.state_dim, hidden_units=hidden_units, gate=config.gate),
      option_body_fn=lambda: FCBody(
          config.state_dim, hidden_units=hidden_units, gate=config.gate),
  )

  config.random_option_prob = LinearSchedule(1.0, 0.1, 1e4)
  config.discount = 0.99
  config.target_network_update_freq = 200
  config.rollout_length = 5
  config.termination_regularizer = 0.01
  config.entropy_weight = 0.01
  config.gradient_clip = 5
  config.max_steps = 1e9
  config.rollout_length = 2048
  config.beta_reg = 0.01
  config.log_interval = 2048
  config.save_interval = 100
  run_steps(OCAgent(config))


# PPO
def ppo_feature(**kwargs):
  generate_tag(kwargs)
  kwargs.setdefault('log_level', 0)
  config = Config()
  config.merge(kwargs)

  config.num_workers = 5
  config.task_fn = lambda: Task(config.game, num_envs=config.num_workers)
  config.eval_env = Task(config.game)
  config.optimizer_fn = lambda params: torch.optim.RMSprop(params, 0.001)
  config.network_fn = lambda: CategoricalActorCriticNet(
      config.state_dim, config.action_dim, FCBody(config.state_dim))
  config.discount = 0.99
  config.use_gae = True
  config.gae_tau = 0.95
  config.entropy_weight = 0.01
  config.gradient_clip = 5
  config.rollout_length = 128
  config.optimization_epochs = 10
  config.mini_batch_size = 32 * 5
  config.ppo_ratio_clip = 0.2
  config.log_interval = 128 * 5 * 10
  run_steps(PPOAgent(config))


def ppo_pixel(**kwargs):
  generate_tag(kwargs)
  kwargs.setdefault('log_level', 0)
  config = Config()
  config.merge(kwargs)

  config.task_fn = lambda: Task(config.game, num_envs=config.num_workers)
  config.eval_env = Task(config.game)
  config.num_workers = 8
  config.optimizer_fn = lambda params: torch.optim.RMSprop(
      params, lr=0.00025, alpha=0.99, eps=1e-5)
  config.network_fn = lambda: CategoricalActorCriticNet(
      config.state_dim, config.action_dim, NatureConvBody())
  config.state_normalizer = ImageNormalizer()
  config.reward_normalizer = SignNormalizer()
  config.discount = 0.99
  config.use_gae = True
  config.gae_tau = 0.95
  config.entropy_weight = 0.01
  config.gradient_clip = 0.5
  config.rollout_length = 128
  config.optimization_epochs = 3
  config.mini_batch_size = 32 * 8
  config.ppo_ratio_clip = 0.1
  config.log_interval = 128 * 8
  config.max_steps = int(2e7)
  run_steps(PPOAgent(config))


def ppo_continuous(**kwargs):
  generate_tag(kwargs)
  kwargs.setdefault('log_level', 0)
  config = Config()
  config.merge(kwargs)

  config.task_fn = lambda: Task(config.game)
  config.eval_env = config.task_fn()

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
  config.max_steps = 1e6
  config.state_normalizer = MeanStdNormalizer()
  run_steps(PPOAgent(config))


# PPOC
def ppoc_continuous(**kwargs):
  generate_tag(kwargs)
  kwargs.setdefault('log_level', 0)
  kwargs.setdefault('learning', 'all')
  kwargs.setdefault('tasks', False)
  config = Config()
  config.merge(kwargs)

  config.num_workers = 8
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
  config.network_fn = lambda: OptionGaussianActorCriticNet(
      config.state_dim,
      config.action_dim,
      num_options=config.num_o,
      phi_body=DummyBody(config.state_dim),
      actor_body=FCBody(
          config.state_dim, hidden_units=hidden_units, gate=config.gate),
      critic_body=FCBody(
          config.state_dim, hidden_units=hidden_units, gate=config.gate),
      option_body_fn=lambda: FCBody(
          config.state_dim, hidden_units=hidden_units, gate=config.gate),
  )
  # config.hid_dim = 64
  # config.network_fn = lambda: LstmOptionGaussianActorCriticNet(
  #     config.state_dim,
  #     config.action_dim,
  #     num_options=config.num_o,
  #     hid_dim=config.hid_dim,
  #     phi_body=DummyBody(config.state_dim),
  #     actor_body=FCBody(
  #         config.state_dim, hidden_units=hidden_units, gate=config.gate),
  #     critic_body=FCBody(
  #         config.state_dim, hidden_units=hidden_units, gate=config.gate),
  #     option_body_fn=lambda: FCBody(
  #         config.state_dim, hidden_units=hidden_units, gate=config.gate),
  # )
  config.optimizer_fn = lambda params: torch.optim.Adam(params, 3e-4, eps=1e-5)
  config.gradient_clip = 0.5

  config.discount = 0.99
  config.rollout_length = 2048
  config.max_steps = 1e9
  config.state_normalizer = MeanStdNormalizer()
  config.log_interval = 2048

  # PPO params
  # training params
  config.optimization_epochs = 10
  config.mini_batch_size = 64
  # model params
  config.use_gae = False
  config.gae_tau = 0.95
  config.ppo_ratio_clip = 0.2
  config.entropy_weight = 0.01

  # OC params
  config.beta_reg = 0.01
  config.delib_cost = 0.01

  run_steps(PPOCAgent(config))


if __name__ == '__main__':
  # use tuna to profile:
  # python -m cProfile -o program.prof run_ppoc.py
  mkdir('log/oc')
  mkdir('tf_log/oc')
  mkdir('data')
  set_one_thread()
  random_seed()
  # select_device(-1)
  select_device(0)
  env_list = [
      'RoboschoolHopper-v1', 'RoboschoolWalker2d-v1',
      'RoboschoolHalfCheetah-v1', 'RoboschoolAnt-v1', 'RoboschoolHumanoid-v1'
  ]
  # select_device(0)

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
  game = 'BipedalWalkerHardcore-v2'
  game = 'LunarLanderContinuous-v2'
  # oc_continuous(game=game)
  # doc_continuous(game=game)
  # a2c_continuous(game=game)
  # ppo_continuous(game=game)
  # ddpg_continuous(game=game)
  # td3_continuous(game=game)
  ppoc_continuous(game=game)

  game = 'BreakoutNoFrameskip-v4'
  # dqn_pixel(game=game)
  # quantile_regression_dqn_pixel(game=game)
  # categorical_dqn_pixel(game=game)
  # a2c_pixel(game=game)
  # n_step_dqn_pixel(game=game)
  # option_critic_pixel(game=game)
  # ppo_pixel(game=game)
