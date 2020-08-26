# -*- coding: utf-8 -*-
from deep_rl import *


def basic_doe_params():
  config = Config()

  # Net Config
  config.state_normalizer = MeanStdNormalizer()
  config.nhead = 1
  config.dmodel = 40
  config.nlayers = 3
  config.nhid = 50
  config.single_transformer_action_net = True

  # Option Framework
  config.num_o = 4
  config.entropy_weight = 0.01
  config.max_steps = 2e6
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

  # MISC
  config.num_workers = 4
  config.log_interval = config.rollout_length * config.num_workers
  config.log_level = 0
  config.log_analyze_stat = False

  return config


doe_params_dict = {
    'visualize':
        dict(num_workers=1),
    'sbenchmark':
        dict(game='HalfCheetah-v2', entropy_weight=0),
    'benchmark':
        dict(game='HalfCheetah-v2'),
    'save_model_debug':
        dict(save_interval=int(5e4 / 2048) * 2048, log_analyze_stat=True),
    'save_model_paper':
        dict(save_interval=int(1e6 / 2048) * 2048),
    'humanoid-v2':
        dict(
            num_o=16, save_interval=int(5e4 / 2048) * 2048, nhid=64,
            dmodel=128),
}
