# -*- coding: utf-8 -*-
from deep_rl import *


def basic_doe_params():
  config = Config()

  # Net Config
  config.state_normalizer = MeanStdNormalizer()
  config.nhead = 1
  config.dmodel = 40
  config.nlayers = 1
  config.nhid = 50
  # action decoder
  config.hidden_units = (64, 64)
  config.single_transformer_action_net = True

  # Option Framework
  config.num_o = 4
  config.o_entropy_weight = 0.01
  config.a_entropy_weight = 0.0
  config.max_steps = 1.1e6
  config.tasks = False

  # RL PG Common
  config.discount = 0.99
  config.gradient_clip = 0.5
  config.optimizer_fn = lambda params: torch.optim.Adam(params, 3e-4, eps=1e-5)

  # PPO
  config.optimization_epochs = 10
  config.mini_batch_size = 64
  config.use_gae = True
  config.gae_tau = 0.95
  config.ppo_ratio_clip_option_max = 0.2
  config.ppo_ratio_clip_option_min = 0.2
  config.ppo_ratio_clip_action = 0.2

  # DOE Train
  config.rollout_length = 2048
  config.shuffle_train = True
  # cosine weight, use for small dmodel
  config.cos_w = 0.0
  config.delib = 0.0

  # MISC
  config.num_workers = 4
  config.log_interval = config.rollout_length * config.num_workers
  config.log_level = 0
  config.log_analyze_stat = False

  return config


doe_params_dict = {
    'benchmark':
        dict(game='HalfCheetah-v2'),
    'benchmarklog':
        dict(
            game='HalfCheetah-v2',
            save_interval=int(5e4 / 2048) * 2048,
            log_analyze_stat=True,
        ),
    'swimmer':
        dict(
            game='Swimmer-v2',
            num_workers=4,
        ),
    'humanoidstandup': {
        'game': 'HumanoidStandup-v2',
        'num_workers': 1
    },
    'reacher': {
        'game': 'Reacher-v2',
        'num_workers': 1
    },
}
