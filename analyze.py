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
      k_list.append(np.full((pad_len, *ts_nw_v.shape[1:]), -1))
    cat_mat = np.concatenate(k_list, axis=0)
    # [S,W,E] -> [W,S,E]
    cat_mat = cat_mat.transpose(1, 0, 2)
    padded_nw_ts_v_dict[k] = cat_mat

  return nw_ts_v_dict, padded_nw_ts_v_dict, pad_len


def plot_key_figure(nw_ts_v_dict,
                    plot_keys,
                    data_settings_dict=None,
                    plt_settings_dict=None):
  if data_settings_dict is None:
    data_settings_dict = dict()
  if plt_settings_dict is None:
    plt_settings_dict = dict()

  plt_settings_dict['meta'] = dict()

  def get_fg_ax_list(plt_settings_dict, nrow=1, ncol=1, new=False):
    if new:
      fg, ax_list = plt.subplots(nrow, ncol, sharex=True)
      plt_settings_dict['meta']['fg'] = fg
      plt_settings_dict['meta']['ax_list'] = ax_list
    else:
      fg = plt_settings_dict.get('fg', None)
      ax_list = plt_settings_dict.get('ax_list', None)
      if fg is None:
        fg, ax_list = plt.subplots(nrow, ncol, sharex=True)
      if ax_list is None:
        if not fg.axes:
          fg.subplot(nrow, ncol, 1)
        ax_list = fg.axes
      plt_settings_dict['meta']['fg'] = fg
      plt_settings_dict['meta']['ax_list'] = ax_list

  num_workers = nw_ts_v_dict[plot_keys[0]].shape[0]
  get_fg_ax_list(plt_settings_dict, num_workers)
  plt_settings_dict['meta']['nrows'] = num_workers

  def q_o_st_value_fn(nw_ts_v_dict, data_settings=None):
    value_mat = nw_ts_v_dict['q_o_st']
    return value_mat

  def ot_value_fn(nw_ts_v_dict, data_settings=None):
    value_mat = nw_ts_v_dict['ot']
    value_mat = value_mat.squeeze(-1)
    return value_mat

  def ot_plot_fn(value_mat, plt_settings=None, meta_settings=None):
    fg, ax_list, nrows = meta_settings['fg'], meta_settings[
        'ax_list'], meta_settings['nrows']
    fg.suptitle(plt_settings['title'])

    cats = np.unique(value_mat)
    for i, ax in enumerate(ax_list.flat):
      ax = sns.heatmap(
          np.expand_dims(value_mat[i], -1).T,
          vmax=cats.max(),
          vmin=cats.min(),
          cmap=plt_settings['cmap'],
          ax=ax)

    # one colorbar
    # https://stackoverflow.com/questions/28356359/one-colorbar-for-seaborn-heatmaps-in-subplot
    # n_cat = cats.shape[0]
    # colorbar = fg.colorbar(ax)
    # r = colorbar.vmax - colorbar.vmin
    # colorbar.set_ticks(
    #     [colorbar.vmin + r / n_cat * (0.5 + i) for i in range(n_cat)])
    # colorbar.set_ticklabels([str(i) for i in cats])

  key_fn = {'ot': [ot_value_fn, ot_plot_fn]}
  for key in plot_keys:
    if key in key_fn:
      v_fn, plt_fn = key_fn[key]
      data_settings = data_settings_dict.get(key, None)
      value_mat = v_fn(nw_ts_v_dict, data_settings)
      plt_settings = plt_settings_dict.get(key, None)
      plt_fn(value_mat, plt_settings, plt_settings_dict['meta'])
      plt_settings_dict['meta']['fg'].set_size_inches(19.20, 9.83)
      plt.savefig(
          './figures/' + key + plt_settings_dict['flnm'] + '.png', dpi=600)


def plot_file(nm, tsbd_be_end_ts, keys, plot_keys):
  key_ts_nw_v_dict = get_np_ktnv_dict(nm, keys=keys)
  nw_ts_v_dict, padded_nw_ts_v_dict, pad_len = get_nw_ts_v_dict(
      tsbd_be_end_ts, key_ts_nw_v_dict, keys)

  ot_cmap = ["1", "#2ecc71", "#3498db", "255", "#e74c3c"]
  plt_settings_dict = {
      'ot': {
          'title': nm,
          'cmap': ot_cmap
      },
      'flnm': nm,
  }

  plot_key_figure(
      padded_nw_ts_v_dict, plot_keys, plt_settings_dict=plt_settings_dict)


if __name__ == "__main__":
  # list to dict of np array

  def single_trans_action_detached():
    ## 2fe8001 single transformer action net; action back grad to option
    # best
    keys = ['s', 'r', 'm', 'at', 'ot', 'pot_ent', 'q_o_st']
    ## 3 starts better than 4
    # tsbd_be_end_ts = [[916000, 1112000]]
    ## end
    # tsbd_be_end_ts = [[1884000, 2000000]]
    ## 1 switching long option
    tsbd_be_end_ts = [[1160000, 1260000]]
    tsbd_be_end_ts = [[1376000, 1444000]]
    tsbd_be_end_ts = [[1704000, 1806000]]
    tsbd_be_end_ts = [[1200000, 2000000]]
    nm = 'HalfCheetah-v2-gate_Tanh()-num_workers_4-remark_TaTrue_EvenAll_SepPPO_Shuffle_DOE_nhead4_dm40_nl1_nhid50-tasks_False-run-41-200803-112825.pkl'
    plot_file(nm, tsbd_be_end_ts, keys, plot_keys=keys)
    nm = 'HalfCheetah-v2-gate_Tanh()-num_workers_4-remark_TaTrue_EvenAll_SepPPO_Shuffle_DOE_nhead4_dm40_nl1_nhid50-tasks_False-run-41-200803-112829.pkl'
    plot_file(nm, tsbd_be_end_ts, keys, plot_keys=keys)
    # worst
    nm = 'HalfCheetah-v2-gate_Tanh()-num_workers_4-remark_TaTrue_EvenAll_SepPPO_Shuffle_DOE_nhead4_dm40_nl1_nhid50-tasks_False-run-41-200803-112831.pkl'
    plot_file(nm, tsbd_be_end_ts, keys, plot_keys=keys)
    nm = 'HalfCheetah-v2-gate_Tanh()-num_workers_4-remark_TaTrue_EvenAll_SepPPO_Shuffle_DOE_nhead4_dm40_nl1_nhid50-tasks_False-run-41-200803-112835.pkl'
    plot_file(nm, tsbd_be_end_ts, keys, plot_keys=keys)
    plt.show()

  def single_trans_action_not_detached():
    ## 2fe8001 single transformer action net; action back grad to option
    # best
    keys = ['s', 'r', 'm', 'at', 'ot', 'pot_ent', 'q_o_st']
    tsbd_be_end_ts = [[0, 100000], [124000, 240000], [1488000, 1640000]]
    nm = 'HalfCheetah-v2-gate_Tanh()-num_workers_4-remark_TaTrue_EvenAll_SepPPO_Shuffle_DOE_nhead4_dm40_nl2_nhid50-tasks_False-run-44-200803-002542.pkl'
    plot_file(nm, tsbd_be_end_ts, keys, plot_keys=keys)
    # worst
    nm = 'HalfCheetah-v2-gate_Tanh()-num_workers_4-remark_TaTrue_EvenAll_SepPPO_Shuffle_DOE_nhead4_dm40_nl2_nhid50-tasks_False-run-44-200803-002545.pkl'
    plot_file(nm, tsbd_be_end_ts, keys, plot_keys=keys)
    plt.show()

  def ppo_adv_option():
    keys = ['s', 'r', 'm', 'at', 'ot', 'pot_ent', 'q_ot_st']
    ## ebcba61 OptPPO use PPO adv for options
    # best 1,2
    tsbd_be_end_ts = [[860000, 984000], [1016000, 1124000], [1848000, 2000000]]
    nm = 'HalfCheetah-v2-gate_Tanh()-num_workers_4-remark_OptPPO_DOE_nhead4_dm100_nl3_nhid50-tasks_False-run-13-200801-012038.pkl'
    plot_file(nm, tsbd_be_end_ts, keys, plot_keys=keys)
    nm = 'HalfCheetah-v2-gate_Tanh()-num_workers_4-remark_OptPPO_DOE_nhead4_dm100_nl3_nhid50-tasks_False-run-13-200801-012043.pkl'
    plot_file(nm, tsbd_be_end_ts, keys, plot_keys=keys)
    # worst 3,4
    nm = 'HalfCheetah-v2-gate_Tanh()-num_workers_4-remark_OptPPO_DOE_nhead4_dm100_nl3_nhid50-tasks_False-run-13-200801-012045.pkl'
    plot_file(nm, tsbd_be_end_ts, keys, plot_keys=keys)
    nm = 'HalfCheetah-v2-gate_Tanh()-num_workers_4-remark_OptPPO_DOE_nhead4_dm100_nl3_nhid50-tasks_False-run-13-200801-012043.pkl'
    plot_file(nm, tsbd_be_end_ts, keys, plot_keys=keys)

  def qloss_use_qot():
    keys = ['s', 'r', 'm', 'at', 'ot', 'pot_ent', 'q_ot_st']
    ## c21ad7b Qbody QotLoss
    tsbd_be_end_ts = [[1348000, 1472000]]
    nm = 'HalfCheetah-v2-gate_Tanh()-num_workers_4-remark_Qbody_QotLoss_DOE_nhead4_dm100_nl3_nhid50-tasks_False-run-13-200801-114225.pkl'
    plot_file(nm, tsbd_be_end_ts, keys, plot_keys=keys)
    nm = 'HalfCheetah-v2-gate_Tanh()-num_workers_4-remark_Qbody_DOE_nhead4_dm100_nl3_nhid50-tasks_False-run-13-200801-113910.pkl'
    plot_file(nm, tsbd_be_end_ts, keys, plot_keys=keys)
    plt.show()

  def initial_run():
    keys = ['s', 'r', 'm', 'at', 'ot', 'pot_ent', 'q_ot_st']
    ## initial run git:1e48131
    # performance up->down
    # best
    nm = 'HalfCheetah-v2-gate_Tanh()-num_workers_4-remark_ShareVnet_DOE_nhead4_dm100_nl3_nhid50-tasks_False-run-4-200731-172620'
    tsbd_be_end_ts = [[570000, 630000], [628000, 652000], [904000, 972000]]
    plot_file(nm, tsbd_be_end_ts, keys, plot_keys=keys)
    plt.show()
    # worst
    nm = 'HalfCheetah-v2-gate_Tanh()-num_workers_4-remark_ShareVnet_DOE_nhead4_dm100_nl3_nhid50-tasks_False-run-4-200731-172622'
    tsbd_be_end_ts = [[524000, 556000], [964000, 996000]]
    plot_file(nm, tsbd_be_end_ts, keys, plot_keys=keys)
    plt.show()

    # compare best, mid, worst
    tsbd_be_end_ts = [[1004000, 1084000], [1212000, 1280000]]
    # best
    nm = 'HalfCheetah-v2-gate_Tanh()-num_workers_4-remark_ShareVnet_DOE_nhead4_dm100_nl3_nhid50-tasks_False-run-4-200731-172620'
    plot_file(nm, tsbd_be_end_ts, keys, plot_keys=keys)
    plt.show()
    # mid
    nm = 'HalfCheetah-v2-gate_Tanh()-num_workers_4-remark_ShareVnet_DOE_nhead4_dm100_nl3_nhid50-tasks_False-run-4-200731-172618'
    plot_file(nm, tsbd_be_end_ts, keys, plot_keys=keys)
    plt.show()
    nm = 'HalfCheetah-v2-gate_Tanh()-num_workers_4-remark_ShareVnet_DOE_nhead4_dm100_nl3_nhid50-tasks_False-run-4-200731-172624'
    plot_file(nm, tsbd_be_end_ts, keys, plot_keys=keys)
    plt.show()
    # worst
    nm = 'HalfCheetah-v2-gate_Tanh()-num_workers_4-remark_ShareVnet_DOE_nhead4_dm100_nl3_nhid50-tasks_False-run-4-200731-172622'
    plot_file(nm, tsbd_be_end_ts, keys, plot_keys=keys)
    plt.show()

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
