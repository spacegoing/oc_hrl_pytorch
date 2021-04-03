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
  elif config.game == 'dm-cheetah':
    tasks = ['run', 'backward']
  elif config.game == 'dm-hopper':
    tasks = ['stand', 'hop']
  elif config.game == 'dm-acrobot':
    tasks = ['swingup', 'swingup_sparse']
  elif config.game == 'dm-finger':
    tasks = ['turn_easy', 'turn_hard']
  elif config.game == 'dm-humanoid-w':
    config.game = 'dm-humanoid'
    tasks = ['stand', 'walk']
  elif config.game == 'dm-humanoid-r':
    config.game = 'dm-humanoid'
    tasks = ['stand', 'run']
  elif config.game == 'dm-manipulator':
    tasks = ['bring_ball', 'bring_peg']
  elif config.game == 'dm-quadruped':
    tasks = ['walk', 'run']
  elif config.game == 'dm-stacker':
    tasks = ['stack_2', 'stack_4']
  elif config.game == 'dm-swimmer':
    tasks = ['swimmer6', 'swimmer15']
  else:
    raise NotImplementedError

  games = ['%s-%s' % (config.game, t) for t in tasks]
  config.tasks = [Task(g, num_envs=config.num_workers) for g in games]
  config.game = games[0]


def wsa(**kwargs):
  config = basic_doe_params()

  config.merge(kwargs)
  config.merge(doe_params_dict.get(kwargs.get('params_set'), dict()))

  if config.tasks:
    set_tasks(config)

  config.task_fn = lambda: Task(config.game, num_envs=config.num_workers)
  config.eval_env = Task(config.game)

  if 'dm-humanoid' in config.game:
    config.nhid = 128

  # assign tag to config
  # # kwargs['remark'] = 'CO_Schedular_r%.2f_UseGae%s_L%.2f_' %\
  # #   (discount, str(use_gae), gae_tau)
  # # kwargs['remark'] = 'CO_Schedular_DOE_nhead%d_dm%d_nl%d_nhid%d' %\
  # #   (nhead, dmodel, nlayers, nhid)
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
  WsaAgent = reload(sys.modules['deep_rl.agent.WSA_agent']).WsaAgent
  run_steps(WsaAgent(config))


if __name__ == "__main__":
  random_seed()
  set_one_thread()
  select_device(-1)
  cf = Config()
  # cf.merge()

  # game = 'HalfCheetah-v2'
  # run = 55
  # tasks = False
  # num_workers = 4
  # gate = nn.Tanh()
  # # OC
  # remark = 'DO_OC'
  # # PPOC
  # remark = 'DO_PPOC'
  # ppoc_continuous(**kwargs)
  # oc_continuous(**kwargs)

  params_set = [
      'dm_cartpole',
      'dm_reacher',
      'dm_fish',
      'dm_cheetah',
      'dm_walker1',
      'dm_walker2',
  ]
  params_set = [
      't_cartpole', 't_reacher', 't_cheetah', 't_fish', 't_walker1', 't_walker2'
  ]
  cf.params_set = params_set[-1]

  ## openai games
  # Valid: half/swimmer:660 walker:663
  # # ffn action; ffn critic
  # cf.run = 222
  # # ffn action; ffn critic; 2 step doe with state_lc (dmodel!=state)
  # cf.run = 111
  # ffn action; ffn critic; 2 step doe; relu(state_lc(obs))
  # cf.run = 660
  # ffn action; ffn critic (mha_st,mha_ot,st); 2 step doe; relu(state_lc(obs));
  # cf.run = 661
  # ffn action; ffn critic (st,ot); 2 step doe; relu(state_lc(obs)); initial flags
  # cf.run = 663
  # walker_small; 10; no cos; orthogonal init embedding
  # cf.run = 669
  # walker dropout=0.0
  # cf.run = 644

  # 700 no self attn; nl3: 1M: 2700-3000-3400  2M: 3500-3700-4200
  # cf.run = 700
  # no self attn; self implemented mha; sate_dim -> dmodel
  # cf.run = 710
  # walker8
  # cf.run = 720
  # delib cost q_o_st[ot_1]+0.01*q_o_st.mean()
  # cf.run = 400
  # mha residule action; dm=10
  # cf.run = 410
  # mha residule action; dm=40
  # cf.run = 420
  # mha residule action; dm=40; num_o=1
  # cf.run = 111
  # ffn action; dm=40; num_o=1; ot=zeros_like(ot)
  # cf.run = 121
  # ffn action; dm=40; num_o=1; ot=ones_like(ot)
  # cf.run = 1211
  # ffn action; dm=40; num_o=4; num_workers=1; cf.params_set = 'walkert'
  # cf.run = 4111
  # No Log; cf.params_set = 'walker'
  # cf.run = 4411
  # delib=0.0; cf.params_set = 'walker'
  # cf.run = 4410

  # 1. state_lc is a must, it projects to skill context vector space
  # 2. cosine similarity works for short dmodel.
  #    large dmodel unlikely to entangle and converges
  #    faster without cosine similarity
  a = [
      'dm-hopper', 'dm-acrobot', 'dm-finger', 'dm-humanoid-w', 'dm-humanoid-r',
      'dm-manipulator', 'dm-quadruped', 'dm-stacker', 'dm-swimmer'
  ]
  cf.params_set = a[8]
  cf.run = 1024
  # DOE
  kwargs = dict(run=cf.run, params_set=cf.params_set)
  wsa(**kwargs)

  # self attn 600 v.s. no self attn 700: nl1 best, std 700>600

  # halfcheetah
  # 700 no self attn; nl3: 1M: 2700-3000-3400  2M: 3500-3700-4200
  # 700 no self attn; nl1: 1M: 2900-3100-3600  2M: 3700-4000-4000
  # 600 self attn; nl1: 1M: 2900-3100-3600  2M: 3700-4000-4000