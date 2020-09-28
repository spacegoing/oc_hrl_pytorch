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
    'visualize':
        dict(num_workers=1),
    'sbenchmark':
        dict(game='HalfCheetah-v2', dmodel=20),
    'benchmark':
        dict(game='HalfCheetah-v2'),
    'benchmarklog':
        dict(
            game='HalfCheetah-v2',
            save_interval=int(5e4 / 2048) * 2048,
            log_analyze_stat=True,
        ),
    'lbenchmark':
        dict(game='HalfCheetah-v2', nlayers=3),
    'swimmer4':
        dict(
            game='Swimmer-v2',
            num_workers=4,
        ),
    'swimmer':
        dict(
            game='Swimmer-v2',
            num_workers=1,
        ),
    # 'humanoidstandup4': {
    #     'game': 'HumanoidStandup-v2',
    #     'num_workers': 4
    # },
    'humanoidstandup': {
        'game': 'HumanoidStandup-v2',
        'num_workers': 1
    },
    # 'reacher4': {
    #     'game': 'Reacher-v2',
    #     'num_workers': 4
    # },
    'reacher': {
        'game': 'Reacher-v2',
        'num_workers': 1
    },
    'walker':
        dict(game='Walker2d-v2', num_workers=1),
    'walker4':
        dict(game='Walker2d-v2', num_workers=4),
    'hopper4':
        dict(
            game='Hopper-v2',
            num_workers=4,
        ),
    'hopper':
        dict(
            game='Hopper-v2',
            num_workers=1,
        ),
    # 'ant4': {
    #     'game': 'Ant-v2',
    #     'num_workers': 4
    # },
    'ant': {
        'game': 'Ant-v2',
        'num_workers': 1
    },
    'humanoid4': {
        'game': 'Humanoid-v2',
        'num_workers': 4
    },
    # 'humanoid': {
    #     'game': 'Humanoid-v2',
    #     'num_workers': 1
    # },
    # 'inverteddoublependulum4': {
    #     'game': 'InvertedDoublePendulum-v2',
    #     'num_workers': 4
    # },
    'inverteddoublependulum': {
        'game': 'InvertedDoublePendulum-v2',
        'num_workers': 1
    },
    'invertedpendulum4': {
        'game': 'InvertedPendulum-v2',
        'num_workers': 4
    },
    # 'invertedpendulum': {
    #     'game': 'InvertedPendulum-v2',
    #     'num_workers': 1
    # },
    'walkerlog':
        dict(
            game='Walker2d-v2',
            num_workers=1,
            delib=0.0,
            save_interval=int(5e4 / 2048) * 2048,
            log_analyze_stat=True,
        ),
    'walkert':
        dict(
            game='Walker2d-v2',
            num_workers=1,
            delib=0.01,
            save_interval=int(5e4 / 2048) * 2048,
            log_analyze_stat=True),
    'walkerq':
        dict(
            game='Walker2d-v2',
            delib=0.0,
            num_o=1,
            # save_interval=int(5e4 / 2048) * 2048,
            # log_analyze_stat=True,
        ),
    'walkerd':
        dict(
            game='Walker2d-v2',
            save_interval=int(5e4 / 2048) * 2048,
            log_analyze_stat=True),
    'walker8':
        dict(
            game='Walker2d-v2',
            num_o=8,
            save_interval=int(5e4 / 2048) * 2048,
            log_analyze_stat=True),
    'walker_large':
        dict(game='Walker2d-v2', nlayers=3),
    'walker_small':
        dict(game='Walker2d-v2', dmodel=10, cos_w=0.5),
    'hopper':
        dict(game='Hopper-v2'),
    'dm_cartpole':
        dict(game='dm-cartpole-balance'),
    'dm_reacher':
        dict(game='dm-reacher-easy'),
    'dm_fish':
        dict(game='dm-fish-upright'),
    'dm_cheetah':
        dict(game='dm-cheetah-run'),
    'dm_walker1':
        dict(game='dm-walker-squat'),
    't_cartpole':
        dict(game='dm-cartpole-b', tasks=True),
    't_reacher':
        dict(game='dm-reacher', tasks=True),
    't_fish':
        dict(game='dm-fish', tasks=True),
    't_cheetah':
        dict(game='dm-cheetah', tasks=True),
    't_walker1':
        dict(game='dm-walker-1', tasks=True),
    't_walker2':
        dict(game='dm-walker-2', tasks=True),
}
