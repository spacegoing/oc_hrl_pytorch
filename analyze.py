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
          cmap=plt_settings['cmap'][:len(cats)],
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

  # ot_cmap = ["1", "#2ecc71", "#3498db", "255", "#e74c3c"]
  ot_cmap = [
      "1", "#E41A1C", "#377EB8", "255", "#4DAF4A", "#984EA3", "#FF7F00",
      "#FFFF33", "#A55628", "#F781BF"
  ]
  plt_settings_dict = {
      'ot': {
          'title': nm,
          'cmap': ot_cmap
      },
      'flnm': nm + str(tsbd_be_end_ts),
  }

  plot_key_figure(
      padded_nw_ts_v_dict, plot_keys, plt_settings_dict=plt_settings_dict)


if __name__ == "__main__":
  # list to dict of np array
  def skill_policy_walker():
    keys = ['s', 'r', 'm', 'at', 'ot', 'pot_ent', 'q_o_st']
    tsbd_be_end_ts = [[1200000, 1400000]]
    nm = 'Walker2d-v2-params_set_walker8-remark_Param_walker8_Net_nhead1_dm40_nl1_nhid50_nO_8-run-710-200917-130941.pkl'
    plot_file(nm, tsbd_be_end_ts, keys, plot_keys=keys)
    nm = 'Walker2d-v2-params_set_walker8-remark_Param_walker8_Net_nhead1_dm40_nl1_nhid50_nO_8-run-710-200917-130951.pkl'
    plot_file(nm, tsbd_be_end_ts, keys, plot_keys=keys)
    tsbd_be_end_ts = [[1200000, 1400000]]
    nm = 'Walker2d-v2-params_set_walker8-remark_Param_walker8_Net_nhead1_dm40_nl1_nhid50_nO_8-run-710-200917-130951.pkl'
    plot_file(nm, tsbd_be_end_ts, keys, plot_keys=keys)
    tsbd_be_end_ts = [[1600000, 1720000]]
    nm = 'Walker2d-v2-params_set_walker8-remark_Param_walker8_Net_nhead1_dm40_nl1_nhid50_nO_8-run-710-200917-130949.pkl'
    plot_file(nm, tsbd_be_end_ts, keys, plot_keys=keys)

  def delib_walker():
    keys = ['s', 'r', 'm', 'init', 'at', 'ot', 'po_t', 'q_o_st']
    tsbd_be_end_ts = [[550000, 580000]]
    lnm = 'Walker2d-v2-params_set_walkerd-remark_Param_walkerd_Net_nhead1_dm40_nl1_nhid50_nO_4-run-420-200918-172616.pkl'
    lsteps_dict_list = read_steps_dict_list(lnm, fpath='./analyze/')
    plot_file(lnm, tsbd_be_end_ts, keys, plot_keys=keys)
    hnm = 'Walker2d-v2-params_set_walkerd-remark_Param_walkerd_Net_nhead1_dm40_nl1_nhid50_nO_4-run-420-200918-172620.pkl'
    hsteps_dict_list = read_steps_dict_list(hnm, fpath='./analyze/')
    plot_file(hnm, tsbd_be_end_ts, keys, plot_keys=keys)

  def show_q_o_st():
    # No delib
    nm = 'Walker2d-v2-params_set_walker8-remark_Param_walker8_Net_nhead1_dm40_nl1_nhid50_nO_8-run-710-200917-130951.pkl'
    nsteps_dict_list = read_steps_dict_list(nm, fpath='./analyze/')

    # delib
    def get_dict(nm, tsbd_be_end_ts, keys):
      key_ts_nw_v_dict = get_np_ktnv_dict(nm, keys=keys)
      nw_ts_v_dict, padded_nw_ts_v_dict, pad_len = get_nw_ts_v_dict(
          tsbd_be_end_ts, key_ts_nw_v_dict, keys)
      return nw_ts_v_dict, padded_nw_ts_v_dict, pad_len

    tsbd_be_end_ts = [[550000, 580000]]
    keys = ['init', 'ot', 'po_t', 'q_o_st']
    lnm = 'Walker2d-v2-params_set_walkerd-remark_Param_walkerd_Net_nhead1_dm40_nl1_nhid50_nO_4-run-420-200918-172616.pkl'
    lnw_ts_v_dict, lpadded_nw_ts_v_dict, lpad_len = get_dict(
        lnm, tsbd_be_end_ts, keys)
    hnm = 'Walker2d-v2-params_set_walkerd-remark_Param_walkerd_Net_nhead1_dm40_nl1_nhid50_nO_4-run-420-200918-172620.pkl'
    hsteps_dict_list = read_steps_dict_list(hnm, fpath='./analyze/')
    hnw_ts_v_dict, hpadded_nw_ts_v_dict, hpad_len = get_dict(
        hnm, tsbd_be_end_ts, keys)

    def print_step(nw_ts_v_dict, i):
      print(nw_ts_v_dict['ot'][:, i - 1, :])
      print(nw_ts_v_dict['po_t'][:, i - 1, :])
      print(nw_ts_v_dict['q_o_st'][:, i, :])
      print(nw_ts_v_dict['init'][:, i, :])

    i = 0
    # delib
    print('low_delib')
    print_step(lnw_ts_v_dict, i)
    # print('high_delib')
    # print_step(hnw_ts_v_dict, i)
    i += 1
