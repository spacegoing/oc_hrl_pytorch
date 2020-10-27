from deep_rl import *
from importlib import reload
import sys
import numpy as np
import pandas as pd
from PIL import Image
import seaborn as sns
import matplotlib.pyplot as plt
from pymongo import MongoClient
import matplotlib
matplotlib.rcParams['savefig.transparent'] = False
sns.set(style="ticks", color_codes=True)

client = MongoClient('mongodb://localhost:27017')
db = client['sa']
# HalfCheetah up:good down: bad
# nm = 'HalfCheetah-v2-params_set_benchmarklog-remark_Param_benchmarklog_Net_nhead1_dm40_nl1_nhid50_nO_4-run-4660-201017-012352-66'
nm = 'HalfCheetah-v2-params_set_benchmarklog-remark_Param_benchmarklog_Net_nhead1_dm40_nl1_nhid50_nO_4-run-4660-201017-012353-347'
# HumanoidStandup
nm = 'HumanoidStandup-v2-params_set_humanoidstanduplog-remark_Param_humanoidstanduplog_Net_nhead1_dm40_nl1_nhid50_nO_4-run-4000-201018-182123-513'
# nm = 'HumanoidStandup-v2-params_set_humanoidstanduplog-remark_Param_humanoidstanduplog_Net_nhead1_dm40_nl1_nhid50_nO_4-run-4000-201018-182123-456'
# Ant
nm = 'Ant-v2-params_set_antlog-remark_Param_antlog_Net_nhead1_dm40_nl1_nhid50_nO_4-run-4000-201018-182123-835'
nm = 'Ant-v2-params_set_antlog-remark_Param_antlog_Net_nhead1_dm40_nl1_nhid50_nO_4-run-4000-201018-182123-367'
# 4002
nm = 'HumanoidStandup-v2-params_set_humanoidstanduplog-remark_Param_humanoidstanduplog_Net_nhead1_dm40_nl1_nhid50_nO_4-run-4002-201019-230420-859'
col = db[nm]

num_workers = 4
num_o = 4
episode_len = 1000


def count_ser_duration(ser):
  '''
  Calculate duration
  ser = [1,1,1,2,2,3,2,2,2]
  duration_dict = {0: [],
                   1: [3],
                   2: [2,3],
                   3: [1]}
  '''
  duration_dict = {k: [] for k in range(num_o)}
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


def plot_duration():
  '''
  Figure 5
  '''
  # rollout_length = 2048, last episode is 512
  res_list = list(col.find({}, {"_id": 0, "ot": 1, 'sim_state': 1, 'r': 1}))
  key_mat_dict = {k: [] for k in res_list[0]}
  for res in res_list:
    for k in res:
      key_mat_dict[k].append(res[k])
  ot_mat = np.array(key_mat_dict['ot']).squeeze(-1)
  sim_state_mat = np.array(key_mat_dict['sim_state'])
  r_mat = np.array(key_mat_dict['r']).squeeze(-1)

  # worker_epi_avg_dict = {k: [] for k in range(num_workers)}
  # worker_duration_dict = dict()
  # for w in range(num_workers):
  # for w in worker_epi_avg_dict:
  #   worker_epi_avg_dict[w] = np.array(worker_epi_avg_dict[w])
  w = 0
  sim_state_mat = sim_state_mat
  ot_ser = ot_mat[:, w]
  r_ser = r_mat[:, w]

  # # episode by config.rollout_length
  # epi_idx_list = list(range(0, sim_state_mat.shape[0], episode_len))
  # episode by environment
  # sim_state_mat = env.sim.get_state().flatten(); [:,0] is timestep
  epi_idx_list = np.where(sim_state_mat[:, 0] == 0)[0].tolist()
  # append last idx; prepend 0 if not there
  epi_idx_list.append(sim_state_mat.shape[0])
  if epi_idx_list[0] != 0:
    epi_idx_list = [0] + epi_idx_list

  def comp_episodic_reward(epi_idx_list, r_ser):
    # average reward in each episode
    epi_reward_list = []
    for i in range(len(epi_idx_list) - 1):
      be, en = epi_idx_list[i], epi_idx_list[i + 1]
      epi_r = r_ser[be:en].sum()
      epi_reward_list.append(epi_r)
    epi_reward_ser = np.array(epi_reward_list)
    return epi_reward_ser

  def comp_episodic_duration(epi_idx_list, ot_ser):
    '''
    average skill duration in each episode
    epi_avg_mat: [num_episodes, num_o]
      num_episodes = len(epi_idx_list)
    '''
    epi_avg_list = []
    for i in range(len(epi_idx_list) - 1):
      be, en = epi_idx_list[i], epi_idx_list[i + 1]
      epi_ser = ot_ser[be:en]
      duration_dict = count_ser_duration(epi_ser)
      epi_avg = np.zeros(num_o)
      for o in range(num_o):
        if duration_dict[o]:
          avg_duration = sum(duration_dict[o]) / len(duration_dict[o])
          epi_avg[o] = avg_duration
      epi_avg_list.append(epi_avg)
    epi_avg_mat = np.array(epi_avg_list)
    return epi_avg_mat

  def viz_episode(episode_idx, game, epi_idx_list, sim_state_mat, ot_ser):
    epi_sim_state = sim_state_mat[
        epi_idx_list[episode_idx]:epi_idx_list[episode_idx + 1]]
    epi_ot = ot_ser[epi_idx_list[episode_idx]:epi_idx_list[episode_idx + 1]]

    mask = [
        [1, 0, 0],  # red
        [0, 1, 0],  # green
        [0, 0, 1],  # blue
        [1, 1, 0],  # yellow
    ]
    task = Task(game)
    env = task.env.envs[0].env
    env.reset()
    gif_frames = []
    for i in range(epi_sim_state.shape[0]):
      env.sim.set_state_from_flattened(epi_sim_state[i])
      env.sim.forward()
      obs = env.render(mode='rgb_array')
      o = epi_ot[i]
      obs = obs * mask[o]
      # gym/mujoco_env flips obs upside-down, here flip it back
      # https://github.com/openai/gym/blob/38a1f630dc9815a567aaf299ae5844c8f8b9a6fa/gym/envs/mujoco/mujoco_env.py#L150
      obs = obs[::-1, :, :]
      img = Image.fromarray(obs.astype(np.uint8))
      gif_frames.append(img)

    gif_frames[0].save(
        './EpisodicGifs/%s_episode_%d.gif' %
        (nm[nm.find('run-'):], episode_idx),
        format='GIF',
        append_images=gif_frames[1:],
        save_all=True,
        duration=epi_sim_state.shape[0] * 0.3,
        loop=0)
    mkdir('./EpisodicGifs/%s_episode_%d/' % (nm[nm.find('run-'):], episode_idx))
    for i, img in enumerate(gif_frames):
      img.save('./EpisodicGifs/%s_episode_%d/step_%d.png' %
               (nm[nm.find('run-'):], episode_idx, i))

  epi_reward_ser = comp_episodic_reward(epi_idx_list, r_ser)
  epi_avg_mat = comp_episodic_duration(epi_idx_list, ot_ser)

  episode_idx = 3400
  game = 'HalfCheetah-v2'
  game = 'Ant-v2'
  game = 'HumanoidStandup-v2'
  viz_episode(episode_idx, game, epi_idx_list, sim_state_mat, ot_ser)

  be = 200
  # clip the last episode, it might not be entire episode
  end = -1
  plot_xticks_flag = True

  cmap = [
      "#E41A1C", "#377EB8", "#4DAF4A", "#984EA3", "#FF7F00", "#FFFF33",
      "#A55628", "#F781BF"
  ]
  plt.figure(figsize=[15, 5 * num_workers])
  for w in worker_epi_avg_dict:
    ts_duration = worker_epi_avg_dict[w][be:end]
    epi_reward_ser = epi_reward_mat[w][be:end]
    plt.subplot(num_workers, 1, w + 1)
    plt.grid()
    plt.title('Workers: %d' % w)
    for i in range(num_o):
      plt.plot(ts_duration[:, i], color=cmap[i], label='Skill %d' % i)
    if w == 0:
      plt.legend()
    ax = plt.gca()
    if plot_xticks_flag:
      xticks = list(range(worker_epi_avg_dict[w].shape[0]))[be:end]
      xticklabels = [str(i) for i in xticks]
      ax.set_xticks(range(len(xticks)))
      ax.set_xticklabels(xticklabels, rotation=90)
    axt = ax.twinx()
    axt.plot(epi_reward_ser, 'y-.', linewidth=2, label='Episodic Reward')
    if w == 0:
      axt.legend(loc=6)
  plt.tight_layout()
  prefix = 's%d_e%d_' % (be, end)
  plt.savefig(
      './figures/' + prefix + nm[nm.find('run-'):] +
      '_skill_duration_epi_return.png',
      dpi=400)


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
