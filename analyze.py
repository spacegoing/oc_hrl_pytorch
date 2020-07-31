import pickle
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
sns.set(style="ticks", color_codes=True)


def read_steps_dict_list(nm, fpath='./analyze/'):
  fnm = fpath + nm
  with open(fnm, 'rb') as f:
    steps_dict_list = pickle.load(f)
  return steps_dict_list


def stack_step_dict_list(steps_dict_list, all_keys):
  # key_ts_nw_v_dict: key step num_workers value
  # key:[S, W, E]
  # total(real))_timestep = S*W; w is batch_size, s is step
  # E: embedding dimension
  key_ts_nw_v_dict = {k: [] for k in all_keys}
  for t in steps_dict_list:
    for k in all_keys:
      tv = t[k]
      key_ts_nw_v_dict[k].append(tv)
  for k in key_ts_nw_v_dict:
    key_ts_nw_v_dict[k] = np.stack(key_ts_nw_v_dict[k])
  return key_ts_nw_v_dict


def get_np_ktnv_dict(nm, fpath='./analyze/', keys=None):
  if not keys:
    keys = ['s', 'r', 'm', 'at', 'ot', 'pot_ent', 'q_ot_st']
  steps_dict_list = read_steps_dict_list(nm, fpath)
  key_ts_nw_v_dict = stack_step_dict_list(steps_dict_list, keys)
  return key_ts_nw_v_dict


def tsbd_ts_bch_step(tsbd_be_end_ts):
  bch_be_end_ts = [[i[0] // 4, i[1] // 4] for i in tsbd_be_end_ts]
  return bch_be_end_ts


def get_nw_ts_v_dict(tsbd_be_end_ts, key_ts_nw_v_dict, keys):
  '''
  Parameters:
    tsbd_be_end_ts: list[[be,end],...]: batch_step * batch_size
      By inspecting tensorboard return images. Which period is interested
    key_ts_nw_v_dict: key:[S, W, E]: key step num_workers value
      E: embedding dimension
      total(real))_timestep = S*W; w is batch_size, s is step

  Returns:
    nw_ts_v_dict: key:[W,S,E]
  '''
  bch_be_end_ts = tsbd_ts_bch_step(tsbd_be_end_ts)
  nw_ts_v_dict = dict()
  for k in keys:
    ts_nw_v = key_ts_nw_v_dict[k]
    k_list = []
    for (be, end) in bch_be_end_ts:
      k_list.append(ts_nw_v[be:end + 1])
    cat_mat = np.concatenate(k_list, axis=0)
    # [S,W,E] -> [W,S,E]
    cat_mat = cat_mat.transpose(1, 0, 2)
    nw_ts_v_dict[k] = cat_mat

  # padded value dict
  total_len = nw_ts_v_dict[keys[0]].shape[1]
  num_seg = len(bch_be_end_ts)
  pad_len = int(total_len / num_seg / 10)
  padded_nw_ts_v_dict = dict()
  for k in keys:
    ts_nw_v = key_ts_nw_v_dict[k]
    k_list = []
    for (be, end) in bch_be_end_ts:
      k_list.append(ts_nw_v[be:end + 1])
      k_list.append(np.full((pad_len, *ts_nw_v.shape[1:]), -2))
    cat_mat = np.concatenate(k_list, axis=0)
    # [S,W,E] -> [W,S,E]
    cat_mat = cat_mat.transpose(1, 0, 2)
    padded_nw_ts_v_dict[k] = cat_mat

  return nw_ts_v_dict, padded_nw_ts_v_dict, pad_len


def plot_key_figure(nw_ts_v_dict, key, key_settings=None, plt_settings=None):

  def ot_value_fn(nw_ts_v_dict, key_settings):
    value_mat = nw_ts_v_dict['ot']
    value_mat = value_mat.squeeze(-1)
    return value_mat

  def ot_plot_fn(value_mat, plt_settings):
    plt.figure()
    ax = plt.axes()
    sns.heatmap(value_mat, cmap=plt_settings['cmap'])
    ax.set_title(plt_settings['title'])
    plt.show()

  key_fn = {'ot': [ot_value_fn, ot_plot_fn]}
  if key in key_fn:
    v_fn, plt_fn = key_fn[key]
    value_mat = v_fn(nw_ts_v_dict, key_settings)
    plt_fn(value_mat, plt_settings)


def plot_file(nm, tsbd_be_end_ts, keys, plot_keys):
  key_ts_nw_v_dict = get_np_ktnv_dict(nm)
  nw_ts_v_dict, padded_nw_ts_v_dict, pad_len = get_nw_ts_v_dict(
      tsbd_be_end_ts, key_ts_nw_v_dict, keys)

  ot_cmap = ["255", "#9b59b6", "#3498db", "#95a5a6", "#e74c3c"]
  plt_settings_dict = {
      'ot': {
          'title': nm,
          'cmap': ot_cmap
      },
  }

  for key in plot_keys:
    plot_key_figure(padded_nw_ts_v_dict, key,\
                    plt_settings=plt_settings_dict.get(key))


if __name__ == "__main__":
  # list to dict of np array
  keys = ['s', 'r', 'm', 'at', 'ot', 'pot_ent', 'q_ot_st']

  nm = 'HalfCheetah-v2-gate_Tanh()-num_workers_4-remark_ShareVnet_DOE_nhead4_dm100_nl3_nhid50-tasks_False-run-4-200731-172620'
  tsbd_be_end_ts = [[570000, 630000], [628000, 652000], [904000, 972000]]
  plot_file(nm, tsbd_be_end_ts, keys, plot_keys=keys)

  nm = 'HalfCheetah-v2-gate_Tanh()-num_workers_4-remark_ShareVnet_DOE_nhead4_dm100_nl3_nhid50-tasks_False-run-4-200731-172622'
  tsbd_be_end_ts = [[524000, 556000], [964000, 996000]]
  plot_file(nm, tsbd_be_end_ts, keys, plot_keys=keys)
# def get_padded_ts_array(tsbd_be_end_ts, pad_len):
#   bch_be_end_ts = tsbd_ts_bch_step(tsbd_be_end_ts)
#   pad_len = 100
#   be = 10
#   ts_padded_array = []
#   for (b, e) in bch_be_end_ts:
#     length = e - b + 1
#     ts_padded_array.append(np.arange(be, be + length))
#     be += length + pad_len
#   ts_padded_array = np.concatenate(ts_padded_array, axis=0)
#   return ts_padded_array
