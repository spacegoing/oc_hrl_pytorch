import sys
from deep_rl import *
import subprocess
from importlib import reload


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

  config.task_fn = lambda: Task(config.game)
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


random_seed(1024)
set_one_thread()
select_device(0)
game = 'HalfCheetah-v2'
run = 40
tasks = False
gate = nn.Tanh()
# PPOC
remark = 'DO_PPOC'
num_workers = 1
# OC
remark = 'DO_OC'
num_workers = 4
kwargs = dict(
    game=game,
    run=run,
    tasks=tasks,
    remark=remark,
    gate=gate,
    num_workers=num_workers)
oc_continuous(**kwargs)
ppoc_continuous(**kwargs)
