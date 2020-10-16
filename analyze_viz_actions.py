from deep_rl import *
from importlib import reload
import sys
import numpy as np
import pandas as pd
from pymongo import MongoClient
import PIL.Image as Image
import matplotlib
matplotlib.rcParams['savefig.transparent'] = True
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style="ticks", color_codes=True)

client = MongoClient('mongodb://localhost:27017')
db = client['sa']
nm = 'HalfCheetah-v2-params_set_benchmarklog-remark_Param_benchmarklog_Net_nhead1_dm40_nl1_nhid50_nO_4-run-4660-200927-095741-94'
col = db[nm]

num_workers = 4
num_o = 4
episode_len = 512
dmodel = 40
perturb_num = 7
perturb_step_size = 2

# Figure 7
random_seed()
set_one_thread()
select_device(-1)
cf = Config()
cf.params_set = 'benchmarklog'

## Load DOE Models
kwargs = dict(
    game='HalfCheetah-v2', run=cf.run, params_set=cf.params_set, nhead=4)

config = basic_doe_params()

config.merge(kwargs)
config.merge(doe_params_dict.get(kwargs.get('params_set'), dict()))

config.task_fn = lambda: Task(config.game, num_envs=config.num_workers)
config.eval_env = Task(config.game)

kwargs['remark'] = 'Param_%s_Net_nhead%d_dm%d_nl%d_nhid%d_nO_%d' %\
  (kwargs.get('params_set',''),
    config.nhead, config.dmodel, config.nlayers, config.nhid,config.num_o)
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
DoeAgent = reload(sys.modules['deep_rl.agent.DOE_agent']).DoeAgent
agent = DoeAgent(config)
task = config.task_fn()

# load model
data_dir = './data/'
fname = 'DoeAgent-HalfCheetah-v2-params_set_benchmarklog-remark_Param_benchmarklog_Net_nhead1_dm40_nl1_nhid50_nO_4-run-4660-983040'
agent.load(data_dir + fname)

## Read states s_t From MongoDB
nm = 'HalfCheetah-v2-params_set_benchmarklog-remark_Param_benchmarklog_Net_nhead1_dm40_nl1_nhid50_nO_4-run-4660-200927-095741-94'
col = db[nm]
state_list = list(
    col.find({}, {
        "_id": 0,
        "s": 1,
        "step": 1
    }).sort([("step", -1)]).limit(1))
states = tensor(np.array(state_list[0]['s']))

## Perturb Dimensions
# list of dmodel length
# each entry is [act_dim, 20]; 20 denotes from -0.1 to 0.9
dim_sample_mean_list = []
for dim in range(config.dmodel):
  sample_mean_list = []
  for i in range(perturb_num):
    with torch.no_grad():
      # ot_hat: [num_workers]
      ot_hat = tensor(np.zeros([4]))
      ot_hat[:] = 3
      ot_hat = ot_hat.long()
      ## beginning of actions part
      # ot: v_t [num_workers, dmodel(embedding size in init)]
      ot = agent.network.embed_option(ot_hat.unsqueeze(0)).detach().squeeze(0)
      ot[:,dim] += -(perturb_step_size * (perturb_num // 2)) +\
        perturb_step_size * i
      # obs_cat: [num_workers, state_dim + dmodel]
      obs = states
      obs_cat = torch.cat([obs, ot], dim=-1)
      # obs_hat: \tilde{S}_t [1, num_workers, dmodel]
      # obs_hat = agent.act_concat_norm(obs_cat)
      obs_hat = obs_cat
      # generate batch inputs for each option
      pat_mean, pat_std = agent.network.act_doe(obs_hat)
      sample_mean_list.append(to_np(pat_mean))
  sample_mean_mat = np.array(sample_mean_list)
  # only keep the first worker's state-action result
  mat = sample_mean_mat[:, 0, :].T
  dim_sample_mean_list.append(mat)

# [dmodel, act_dim, 20]
dim_mean_mat = np.array(dim_sample_mean_list)
# with open('dim_mean_mat.npy', 'wb') as f:
#   np.save(f, dim_mean_mat)


def plot_all_perturbed_dims():
  out_dir = './EnvFigs'

  def save_current_obs(env, dim, perturb_size, repeat):
    obs = env.render(mode='rgb_array')
    # gym/mujoco_env flips obs upside-down, here flip it back
    # https://github.com/openai/gym/blob/38a1f630dc9815a567aaf299ae5844c8f8b9a6fa/gym/envs/mujoco/mujoco_env.py#L150
    obs = obs[::-1, :, :]
    img = Image.fromarray(obs)
    img.save('%s/dim%d_perturb%.2f_%04d.png' %
             (out_dir, dim, perturb_size, repeat))
    return img

  # Plot each perturbed action
  env = task.env.envs[0].env
  # todo: change st to correct state
  env.reset()
  init_st = env.sim.get_state().flatten()

  # for dim in range(dmodel):
  for dim in range(5):
    # Save init state
    for perturb_id in range(perturb_num):
      perturb_size = -(perturb_step_size * (perturb_num // 2)) +\
              perturb_step_size * perturb_id
      env.sim.set_state_from_flattened(init_st)
      env.sim.forward()
      at = dim_mean_mat[dim, :, perturb_id]
      gif_frames = []
      for repeat in range(5):
        img = save_current_obs(env, dim, perturb_size, repeat)
        gif_frames.append(img)
        env.step(at)
      gif_frames[0].save(
          './EnvGifs/dim%d_perturb%.2f.gif' % (dim, perturb_size),
          format='GIF',
          append_images=gif_frames[1:],
          save_all=True,
          duration=500,
          loop=0)


def plot_pertrub_heatmap():
  # dim_mean_mat = np.load('dim_mean_mat.npy')
  for i in range(dmodel):
    plt.figure(figsize=(15, 5))
    mat = dim_mean_mat[i]
    norm = (mat.T - mat.mean(axis=1).T) / mat.std(axis=1).T
    norm = norm.T
    labels = ['bthigh', 'bshin', 'bfoot', 'fthigh', 'fshin', 'ffoot']
    cols = [
        '%.2f' % i for i in [
            -(perturb_step_size * (perturb_num // 2)) + perturb_step_size * p
            for p in range(perturb_num)
        ]
    ]
    norm = pd.DataFrame(norm)
    norm.index = labels
    norm.columns = cols
    ax = sns.heatmap(norm, cbar_kws={"orientation": "horizontal"}, cbar=False)
    # plt.xlabel('Dim %d' % i, fontsize=50, fontweight='bold')
    # ax.get_xaxis().set_ticks([])
    # ax.get_yaxis().set_ticks([])
    ax.set_yticklabels(
        ax.get_yticklabels(), fontsize=20, fontweight='bold', rotation=360)
    ax.set_xticklabels(ax.get_xticklabels(), fontsize=15, rotation=360)
    plt.savefig('./EnvHeatmaps/%d_dim_perturb.png' % i)
    plt.close()


plot_pertrub_heatmap()
plot_all_perturbed_dims()
