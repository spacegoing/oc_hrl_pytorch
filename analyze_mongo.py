import pickle
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pymongo import MongoClient
import traceback
sns.set(style="ticks", color_codes=True)

client = MongoClient('mongodb://localhost:27017')
db = client['sa']
nm = 'HalfCheetah-v2-params_set_benchmarklog-remark_Param_benchmarklog_Net_nhead1_dm40_nl1_nhid50_nO_4-run-4660-200927-095741-94'
col = db[nm]

num_workers = 4
num_o = 4
episode_len = 512
# rollout_length = 2048, last episode is 512
res_list = list(col.find({}, {"_id": 0, "ot": 1}))
ot_list = []
for res in res_list:
  ot_list.append(res['ot'])
ot_mat = np.array(ot_list).squeeze(-1)


def plot_duration():

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

  plt.figure()
  for i in range(num_o):
    plt.plot(ts_duration[:, i], color=cmap[i], label='Skill %d' % i)
  plt.legend()
  plt.show()


def plot_episode_option():
  res_list = list(col.find({}, {"_id": 0, "ot": 1}))
  ot_list = []
  for res in res_list:
    ot_list.append(res['ot'])
  ot_mat = np.array(ot_list).squeeze(-1)
  epi_idx = 449
  start_idx = list(range(0, ot_mat.shape[0], episode_len))[epi_idx]
  value_mat = ot_mat[start_idx + 112:start_idx + 105 + episode_len].T
  fg, ax_list = plt.subplots(num_workers, 1, sharex=True)

  cats = np.array(list(range(4)))
  cmap = [
      "#E41A1C", "#377EB8", "#4DAF4A", "#984EA3", "#FF7F00", "#FFFF33",
      "#A55628", "#F781BF"
  ]

  for i, ax in enumerate(ax_list.flat):
    ax = sns.heatmap(
        np.expand_dims(value_mat[i], -1).T,
        vmax=cats.max(),
        vmin=cats.min(),
        cmap=cmap[:len(cats)],
        ax=ax)
  plt.show()
