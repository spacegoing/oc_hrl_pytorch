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

  # WSA
  config.skill_lag = 1
  config.max_skill_lag = config.rollout_length
  # skill embedding size = [num_o+1, embed]
  # the last one is for padding
  config.padding_mask_token = config.num_o

  # MISC
  config.num_workers = 4
  config.log_interval = config.rollout_length * config.num_workers
  config.log_level = 0
  config.log_analyze_stat = False

  return config


doe_params_dict = {
    'sbenchmark':
        dict(game='HalfCheetah-v2', dmodel=20),
    'benchmark':
        dict(game='HalfCheetah-v2'),
    'benchmarklog':
        dict(
            game='HalfCheetah-v2',
            save_interval=int(1e4 / 2048) * 2048,
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
    'humanoidstandup4': {
        'game': 'HumanoidStandup-v2',
        'num_workers': 4
    },
    'humanoidstandup': {
        'game': 'HumanoidStandup-v2',
        'num_workers': 1
    },
    'humanoidstanduplog': {
        'game': 'HumanoidStandup-v2',
        'num_workers': 1,
        'log_analyze_stat': True,
    },
    'reacher4': {
        'game': 'Reacher-v2',
        'num_workers': 4
    },
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
    'ant4': {
        'game': 'Ant-v2',
        'num_workers': 4
    },
    'ant': {
        'game': 'Ant-v2',
        'num_workers': 1
    },
    'antlog': {
        'game': 'Ant-v2',
        'num_workers': 1,
        'log_analyze_stat': True,
    },
    'humanoid4': {
        'game': 'Humanoid-v2',
        'num_workers': 4
    },
    'humanoid': {
        'game': 'Humanoid-v2',
        'num_workers': 1
    },
    'inverteddoublependulum7': {
        'game': 'InvertedDoublePendulum-v2',
        'num_workers': 7
    },
    'inverteddoublependulum4': {
        'game': 'InvertedDoublePendulum-v2',
        'num_workers': 4
    },
    'inverteddoublependulum': {
        'game': 'InvertedDoublePendulum-v2',
        'num_workers': 1
    },
    'invertedpendulum4': {
        'game': 'InvertedPendulum-v2',
        'num_workers': 4
    },
    'invertedpendulum': {
        'game': 'InvertedPendulum-v2',
        'num_workers': 1
    },
    'dmtcartpole':
        dict(
            game='dm-cartpole-b',
            tasks=True,
            max_steps=2e6,
            num_workers=4,
            dmodel=20),
    'dmtreacher':
        dict(
            game='dm-reacher',
            tasks=True,
            max_steps=2e6,
            num_workers=4,
            dmodel=20),
    'dmtfish':
        dict(
            game='dm-fish', tasks=True, max_steps=2e6, num_workers=4,
            dmodel=20),
    'dmtcheetah':
        dict(
            game='dm-cheetah',
            tasks=True,
            max_steps=2e6,
            num_workers=4,
            dmodel=20),
    'dmtwalker':
        dict(
            game='dm-walker',
            tasks=True,
            max_steps=2e6,
            num_workers=4,
            dmodel=20),
    'dmtwalker1':
        dict(
            game='dm-walker-1',
            tasks=True,
            max_steps=2e6,
            num_workers=4,
            dmodel=20),
    'dmtwalker2':
        dict(
            game='dm-walker-2',
            tasks=True,
            max_steps=2e6,
            num_workers=4,
            dmodel=20),
    'dmwalker2_s':
        dict(
            game='dm-walker-2',
            tasks=True,
            max_steps=2e6,
            num_workers=4,
            dmodel=20),
    'dmwalker2_m':
        dict(
            game='dm-walker-2',
            tasks=True,
            max_steps=2e6,
            num_workers=4,
            dmodel=30),
    'dmwalker2_l':
        dict(
            game='dm-walker-2',
            tasks=True,
            max_steps=2e6,
            num_workers=4,
            dmodel=60,
            nhid=80),
    'dm-hopper':
        dict(
            game='dm-hopper',
            tasks=True,
            max_steps=2e6,
            num_workers=4,
            dmodel=20),
    'dm-acrobot':
        dict(
            game='dm-acrobot',
            tasks=True,
            max_steps=2e6,
            num_workers=4,
            dmodel=20),
    'dm-finger':
        dict(
            game='dm-finger',
            tasks=True,
            max_steps=2e6,
            num_workers=4,
            dmodel=20),
    'dm-humanoid-w':
        dict(
            game='dm-humanoid-w',
            tasks=True,
            max_steps=2e6,
            num_workers=4,
            dmodel=20),
    'dm-humanoid-r':
        dict(
            game='dm-humanoid-r',
            tasks=True,
            max_steps=2e6,
            num_workers=4,
            dmodel=20),
    'dm-manipulator':
        dict(
            game='dm-manipulator',
            tasks=True,
            max_steps=2e6,
            num_workers=4,
            dmodel=20),
    'dm-quadruped':
        dict(
            game='dm-quadruped',
            tasks=True,
            max_steps=2e6,
            num_workers=4,
            dmodel=20),
    'dm-stacker':
        dict(
            game='dm-stacker',
            tasks=True,
            max_steps=2e6,
            num_workers=4,
            dmodel=20),
    'dm-swimmer':
        dict(
            game='dm-swimmer',
            tasks=True,
            max_steps=2e6,
            num_workers=4,
            dmodel=20),
}
