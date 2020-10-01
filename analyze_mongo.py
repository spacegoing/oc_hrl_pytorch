from deep_rl import *
from importlib import reload
import sys
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pymongo import MongoClient
import matplotlib
matplotlib.rcParams['savefig.transparent'] = True
sns.set(style="ticks", color_codes=True)

client = MongoClient('mongodb://localhost:27017')
db = client['sa']
nm = 'HalfCheetah-v2-params_set_benchmarklog-remark_Param_benchmarklog_Net_nhead1_dm40_nl1_nhid50_nO_4-run-4660-200927-095741-94'
col = db[nm]

num_workers = 4
num_o = 4
episode_len = 512


def plot_duration():
  # rollout_length = 2048, last episode is 512
  res_list = list(col.find({}, {"_id": 0, "ot": 1}))
  ot_list = []
  for res in res_list:
    ot_list.append(res['ot'])
  ot_mat = np.array(ot_list).squeeze(-1)

  # Figure 5
  def count_ser_duration(ser, duration_dict):
    init = ser[0]
    counter = {k: 0 for k in range(num_o)}
    counter[init] += 1
    for t in range(1, ser.shape[0]):
      mask = init == ser[t]
      if mask:
        counter[init] += 1
      else:
        duration_dict[init].append(counter[init])
        counter[init] = 0
        counter[ser[t]] = 1
        init = ser[t]
    if mask:
      duration_dict[init].append(counter[init])
    else:
      duration_dict[ser[-1]].append(1)
    return duration_dict

  epi_avg_list = []
  for i in range(0, ot_mat.shape[0], episode_len):
    epi_mat = ot_mat[i:i + episode_len]
    duration_dict = {k: [] for k in range(num_o)}
    for w in range(num_workers):
      ser = epi_mat[:, w]
      duration_dict = count_ser_duration(ser, duration_dict)
    epi_avg = np.zeros(num_o)
    for o in range(num_o):
      if duration_dict[o]:
        avg_duration = sum(duration_dict[o]) / len(duration_dict[o])
        epi_avg[o] = avg_duration
    epi_avg_list.append(epi_avg)

  ts_duration = np.array(epi_avg_list)
  cmap = [
      "#E41A1C", "#377EB8", "#4DAF4A", "#984EA3", "#FF7F00", "#FFFF33",
      "#A55628", "#F781BF"
  ]

  plt.figure(figsize=[4, 2])
  for i in range(num_o):
    plt.plot(ts_duration[:, i], color=cmap[i], label='Skill %d' % i)
  plt.legend()
  plt.show()


def plot_episode_option():
  # Figure 6
  res_list = list(col.find({}, {"_id": 0, "ot": 1}))
  ot_list = []
  for res in res_list:
    ot_list.append(res['ot'])
  ot_mat = np.array(ot_list).squeeze(-1)
  epi_idx = 449
  start_idx = list(range(0, ot_mat.shape[0], episode_len))[epi_idx]
  value_mat = ot_mat[start_idx + 112:start_idx + 105 + episode_len].T
  fg, ax_list = plt.subplots(num_workers, 1, sharex=True, sharey=True)
  cbar_ax = fg.add_axes([.91, .3, .03, .4])

  cats = np.array(list(range(4)))
  cmap = [
      "#E41A1C", "#377EB8", "#4DAF4A", "#984EA3", "#FF7F00", "#FFFF33",
      "#A55628", "#F781BF"
  ]

  for i, ax in enumerate(ax_list.flat):
    ax = sns.heatmap(
        np.expand_dims(value_mat[i], -1).T,
        cbar=i == 0,
        cbar_ax=None if i else cbar_ax,
        vmax=cats.max(),
        vmin=cats.min(),
        cmap=cmap[:len(cats)],
        ax=ax)
    ax.set_yticklabels([], fontsize=10, rotation=360, fontweight='bold')
  fg.tight_layout(rect=[0, 0, .9, 1])
  plt.show()


def plot_action_color_map():
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
  env = config.task_fn()

  # load model
  data_dir = './data/'
  fname = 'DoeAgent-HalfCheetah-v2-params_set_benchmarklog-remark_Param_benchmarklog_Net_nhead1_dm40_nl1_nhid50_nO_4-run-4660-983040'
  agent.load(data_dir + fname)
  ## Plot All Skills
  # param_gen = agent.network.named_parameters()
  # param_dict = dict()
  # for n, p in param_gen:
  #   print(n)
  #   print(p)
  #   param_dict[n] = p
  # wt = param_dict['embed_option.weight']
  # wt = to_np(wt)
  # with open('skill_matrix.npy', 'wb') as f:
  #   np.save(f, wt)
  # wt = np.load('skill_matrix.npy')
  # plt.figure(figsize=(50, 10))
  # ax = sns.heatmap(
  #     wt,
  #     cbar_kws={"orientation": "horizontal"},
  #     cbar=False,
  #     cmap='OrRd',
  #     annot=True,
  #     annot_kws={"size": 20})
  # ax.set_yticklabels(['Skill %d' % i for i in range(4)],
  #                    fontsize=40,
  #                    rotation=360,
  #                    fontweight='bold')
  # ax.set_xticklabels(
  #     ax.get_xticklabels(), fontsize=30, rotation=360, fontweight='bold')
  # plt.xlabel('Skill Context Vector Dimensions', fontsize=40, fontweight='bold')
  # plt.savefig('./figures/all_skill_vectors.png')
  # plt.close()

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
  dim_sample_mean_list = []
  for dim in range(config.dmodel):
    sample_mean_list = []
    for i in range(20):
      with torch.no_grad():
        # ot_hat: [num_workers]
        ot_hat = tensor(np.zeros([4]))
        ot_hat[:] = 3
        ot_hat = ot_hat.long()
        ## beginning of actions part
        # ot: v_t [num_workers, dmodel(embedding size in init)]
        ot = agent.network.embed_option(ot_hat.unsqueeze(0)).detach().squeeze(0)
        ot[:, dim] += -0.1 + 0.01 * i
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
    mat = sample_mean_mat[:, 0, :].T
    dim_sample_mean_list.append(mat)

  dim_mean_mat = np.array(dim_sample_mean_list)
  # with open('dim_mean_mat.npy', 'wb') as f:
  #   np.save(f, dim_mean_mat)

  dim_mean_mat = np.load('dim_mean_mat.npy')
  for i in range(40):
    # for i in range(config.dmodel):
    plt.figure(figsize=(15, 5))
    mat = dim_mean_mat[i]
    norm = (mat.T - mat.mean(axis=1).T) / mat.std(axis=1).T
    norm = norm.T
    labels = ['bthigh', 'bshin', 'bfoot', 'fthigh', 'fshin', 'ffoot']
    cols = ['%.2f' % i for i in np.arange(-0.1, 0.1, 0.01)]
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
    plt.savefig('./figures/%d_dim_perturb.png' % i)
    plt.close()
