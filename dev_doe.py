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


# OC
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

  if config.tasks:
    set_tasks(config)

  if 'dm-humanoid' in config.game:
    hidden_units = (128, 128)
  else:
    hidden_units = (64, 64)

  config.task_fn = lambda: Task(config.game, num_envs=config.num_workers)
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
  config.random_option_prob = LinearSchedule(0.1)
  config.optimizer_fn = lambda params: torch.optim.Adam(params, 3e-4, eps=1e-5)
  config.discount = 0.99
  config.gradient_clip = 0.5
  config.rollout_length = 5
  config.beta_reg = 0.01
  config.state_normalizer = MeanStdNormalizer()
  config.target_network_update_freq = int(1e3)

  OCAgent = reload(sys.modules['deep_rl.agent.OC_agent']).OCAgent
  run_steps(OCAgent(config))


# PPOC
def ppoc_continuous(**kwargs):
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

  config.task_fn = lambda: Task(config.game, num_envs=config.num_workers)
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
  config.rollout_length = 2048
  config.optimization_epochs = 10
  config.mini_batch_size = 64
  config.ppo_ratio_clip = 0.2
  config.log_interval = 2048
  config.state_normalizer = MeanStdNormalizer()

  PPOCAgent = reload(sys.modules['deep_rl.agent.PPOC_agent']).PPOCAgent
  run_steps(PPOCAgent(config))


# DOE
def doe_continuous(**kwargs):
  discount = 0.99
  use_gae = True
  gae_tau = 0.95
  shuffle_train = True
  ppo_ratio_clip_option_max = 0.2
  ppo_ratio_clip_option_min = 0.2
  ppo_opt_loss = True
  single_transformer_action_net = True
  # kwargs['remark'] = 'CO_Schedular_r%.2f_UseGae%s_L%.2f_' %\
  #   (discount, str(use_gae), gae_tau)
  nhead = 4
  dmodel = 100
  nlayers = 3
  nhid = 50
  kwargs['remark'] = 'Ta%s_EvenAll_SepPPO_Shuffle_DOE_nhead%d_dm%d_nl%d_nhid%d' %\
    (single_transformer_action_net, nhead, dmodel, nlayers, nhid)
  # kwargs['remark'] = 'CO_Schedular_DOE_nhead%d_dm%d_nl%d_nhid%d' %\
  #   (nhead, dmodel, nlayers, nhid)
  generate_tag(kwargs)
  kwargs.setdefault('log_level', 0)
  kwargs.setdefault('num_o', 4)
  kwargs.setdefault('gate', nn.ReLU())
  kwargs.setdefault('entropy_weight', 0.01)
  kwargs.setdefault('tasks', False)
  kwargs.setdefault('max_steps', 2e6)
  config = Config()
  config.merge(kwargs)
  config.log_analyze_stat = True
  config.ppo_opt_loss = ppo_opt_loss
  config.shuffle_train = shuffle_train
  config.single_transformer_action_net = single_transformer_action_net

  if config.tasks:
    set_tasks(config)

  if 'dm-humanoid' in config.game:
    hidden_units = (128, 128)
  else:
    hidden_units = (64, 64)

  config.task_fn = lambda: Task(config.game, num_envs=config.num_workers)
  config.eval_env = Task(config.game)

  DoeContiOneOptionNet = reload(
      sys.modules['deep_rl.network.network_heads']).DoeContiOneOptionNet
  config.network_fn = lambda: DoeContiOneOptionNet(
      config.state_dim,
      config.action_dim,
      num_options=config.num_o,
      nhead=nhead,
      dmodel=dmodel,
      nlayers=nlayers,
      nhid=nhid,
      dropout=0.2,
      config=config)
  config.optimizer_fn = lambda params: torch.optim.Adam(params, 3e-4, eps=1e-5)
  config.discount = discount
  config.use_gae = use_gae
  config.gae_tau = gae_tau
  config.ppo_ratio_clip_option_max = ppo_ratio_clip_option_max
  config.ppo_ratio_clip_option_min = ppo_ratio_clip_option_min
  config.gradient_clip = 0.5
  config.rollout_length = 2048
  config.optimization_epochs = 10
  config.mini_batch_size = 64
  config.ppo_ratio_clip_action = 0.2
  config.log_interval = config.rollout_length * config.num_workers
  config.state_normalizer = MeanStdNormalizer()

  DoeAgent = reload(sys.modules['deep_rl.agent.DOE_agent']).DoeAgent
  run_steps(DoeAgent(config))


random_seed()
set_one_thread()
select_device(-1)
game = 'HalfCheetah-v2'
run = 44
tasks = False
num_workers = 4
gate = nn.Tanh()
# OC
remark = 'DO_OC'
# PPOC
remark = 'DO_PPOC'
# DOE
remark = 'DO_DOE'
kwargs = dict(
    game=game,
    run=run,
    tasks=tasks,
    remark=remark,
    gate=gate,
    num_workers=num_workers)
doe_continuous(**kwargs)
# ppoc_continuous(**kwargs)
# oc_continuous(**kwargs)
