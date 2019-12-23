#######################################################################
# Copyright (C) 2017 Shangtong Zhang(zhangshangtong.cpp@gmail.com)    #
# Permission given to modify the code as long as you keep this        #
# declaration at the top                                              #
#######################################################################

from deep_rl import *


def show_uncontained_keys(config):
  contained_keys = set(config.__dict__.keys())
  uncontained_keys = dir(config)
  uncontained_keys = {k for k in uncontained_keys if not k.startswith('_')}
  missing_keys = uncontained_keys - contained_keys
  print(missing_keys)


def sort_config_keys(config):
  kv = config.__dict__
  sk = sorted(kv)
  sorted_kv = ['%s: %s' % (k, kv[k]) for k in sk]
  print(sorted_kv)


# Option-Critic
def option_critic_feature(**kwargs):

  # kwargs = {'game': 'CartPole-v0'}

  generate_tag(kwargs)
  kwargs.setdefault('log_level', 0)
  config = Config()
  config.merge(kwargs)

  config.num_workers = 6
  config.task_fn = lambda: Task(config.game, num_envs=config.num_workers)
  config.eval_env = Task(config.game)
  config.optimizer_fn = lambda params: torch.optim.RMSprop(params, 0.001)
  # FCBody: 2 layers FC net with ReLU gate. (4,64 -> 64,64)
  config.network_fn = lambda: OptionCriticNet(
      FCBody(config.state_dim), config.action_dim, num_options=7)
  config.random_option_prob = LinearSchedule(1.0, 0.1, 1e4)
  config.discount = 0.99
  config.target_network_update_freq = 200
  config.rollout_length = 10
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
      params, lr=7e-4, alpha=0.99, eps=1e-5)
  config.network_fn = lambda: OptionCriticNet(
      NatureConvBody(), config.action_dim, num_options=4)
  config.random_option_prob = LinearSchedule(0.1)
  config.state_normalizer = ImageNormalizer()
  config.reward_normalizer = SignNormalizer()
  config.discount = 0.99
  config.target_network_update_freq = 10000
  config.rollout_length = 5
  config.gradient_clip = 5
  config.max_steps = int(10e7)
  config.entropy_weight = 0.01
  config.termination_regularizer = 0.01
  run_steps(OptionCriticAgent(config))


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

  config.num_workers = 8
  config.task_fn = lambda: Task(config.game, num_envs=config.num_workers)
  config.eval_env = Task(config.game)

  config.network_fn = lambda: GaussianActorCriticNet(
      config.state_dim,
      config.action_dim,
      actor_body=FCBody(config.state_dim, gate=torch.tanh),
      critic_body=FCBody(config.state_dim, gate=torch.tanh))
  config.optimizer_fn = lambda params: torch.optim.Adam(params, 3e-4, eps=1e-5)
  config.gradient_clip = 0.5

  config.discount = 0.99
  config.rollout_length = 2048
  config.optimization_epochs = 10
  config.mini_batch_size = 64
  config.max_steps = 1e6
  config.state_normalizer = MeanStdNormalizer()
  config.log_interval = 2048

  # PPO Params
  config.use_gae = True
  config.gae_tau = 0.95
  config.ppo_ratio_clip = 0.2

  run_steps(PPOAgent(config))

# Option-Critic
def option_critic_continuous(**kwargs):

  # kwargs = {'game': 'CartPole-v0'}

  generate_tag(kwargs)
  kwargs.setdefault('log_level', 0)
  config = Config()
  config.merge(kwargs)

  config.num_workers = 16
  config.task_fn = lambda: Task(config.game, num_envs=config.num_workers)
  config.eval_env = Task(config.game)

  # FCBody: 2 layers FC net with ReLU gate. (4,64 -> 64,64)
  config.network_fn = lambda: OptionCriticGaussianNet(
      FCBody(config.state_dim), config.action_dim, num_options=2)
  config.optimizer_fn = lambda params: torch.optim.Adam(params, 3e-4, eps=1e-5)
  config.gradient_clip = 0.5
  # config.optimizer_fn = lambda params: torch.optim.RMSprop(params, 0.001)
  # config.gradient_clip = 5

  config.discount = 0.99
  config.rollout_length = 50
  # config.optimization_epochs = 10
  # config.mini_batch_size = 64
  config.max_steps = 1e8
  config.state_normalizer = MeanStdNormalizer()
  config.log_interval = 2

  # OC Params
  # config.reward_normalizer = SignNormalizer()
  config.target_network_update_freq = 200
  config.random_option_prob = LinearSchedule(1.0, 0.1, 1e4)
  config.termination_regularizer = 0.01
  config.entropy_weight = 0.01

  run_steps(OptionCriticContinuousAgent(config))



if __name__ == '__main__':
  mkdir('log')
  mkdir('tf_log')
  set_one_thread()
  random_seed()
  # select_device(-1) # use cpu
  select_device(0)

  game = 'CartPole-v0'
  # dqn_feature(game=game)
  # quantile_regression_dqn_feature(game=game)
  # categorical_dqn_feature(game=game)
  # a2c_feature(game=game)
  # n_step_dqn_feature(game=game)
  # option_critic_feature(game=game)
  # ppo_feature(game=game)

  # game = 'HalfCheetah-v2'
  game = 'Hopper-v2'
  game = 'LunarLanderContinuous-v2'
  game = 'BipedalWalkerHardcore-v2'
  # a2c_continuous(game=game)
  option_critic_continuous(game=game)
  # ppo_continuous(game=game)
  # ddpg_continuous(game=game)
  # td3_continuous(game=game)

  game = 'BreakoutNoFrameskip-v4'
  game = 'MsPacmanNoFrameskip-v0'
  # dqn_pixel(game=game)
  # quantile_regression_dqn_pixel(game=game)
  # categorical_dqn_pixel(game=game)
  # a2c_pixel(game=game)
  # n_step_dqn_pixel(game=game)
  option_critic_pixel(game=game)
  # ppo_pixel(game=game)
