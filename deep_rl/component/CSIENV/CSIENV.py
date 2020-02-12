# -*- coding: utf-8 -*-
import gym
from gym import spaces
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import random
from random import shuffle

MAX_STEPS = 20000

INIT_CASH = 1000000
MIN_TRADE_VOLUME = 100

stamp_tax = 0.001
commission = 0.00015
transfer_tax_sh = 0.00002

wcsi_path = '/Users/spacegoing/Downloads/scp/scp/ppoc/csi300_20150130.csv'
csi_dir = [
    '/Users/spacegoing/Downloads/scp/scp/ppoc/a/',
    '/Users/spacegoing/Downloads/scp/scp/ppoc/b/',
    '/Users/spacegoing/Downloads/scp/scp/ppoc/c/',
]
stock_feat_dim = 6


def fee(amount, mkt='sh'):
  # amount = price * volume
  transfer_fee = 0
  if mkt == 'sh':
    transfer_fee = amount * transfer_tax_sh
  return transfer_fee + amount * commission


def sell_fee(amount, mkt='sh'):
  fix_fee = fee(amount, mkt)
  return fix_fee + amount * stamp_tax


def buy_fee(amount, mkt='sh'):
  return fee(amount, mkt)


class Csi300Env(gym.Env):
  """A stock trading environment for OpenAI gym"""
  metadata = {'render.modes': ['human']}

  def __init__(self, seq_len=1):
    '''

    Attributes:
      stock_df: DataFrame[seq_len, self.stock_dim];
        columns: ohlcva; index: Datetimestamp
      norm_nd: np.ndarray[seq_len, self.stock_dim];
        columns: ohlcva
    '''
    super().__init__()

    self.seq_len = seq_len

    # Actions of the format Trade a[0]<1, Hold a[0]<2
    # target stock_value/total_asset_value = a[1]%
    self.action_space = spaces.Box(
        low=np.array([0, 0]), high=np.array([2, 1]), dtype=np.float16)

    # asset (cash + stock market value), stock value / asset
    self.portfolio_dim = 2
    self.stock_dim = 6  # ohclva
    low_states = np.array([-1] * seq_len * self.stock_dim + [0, 0])
    high_states = np.array([1] * seq_len * self.stock_dim + [np.inf, np.inf])
    self.observation_space = spaces.Box(
        low=low_states, high=high_states, dtype=np.float16)

  def _next_observation(self):
    # Get the stock data points for the last 5 days and scale to between 0-1
    self.stock_df, self.norm_nd, \
      self.done, self.stock_name = self.reader.get_next_step(
    )
    obs = np.append(
        self.norm_nd.reshape(-1), [
            self.total_asset_value / INIT_CASH,
            self.stock_value / self.total_asset_value
        ],
        axis=0)

    return obs

  def _take_action(self, action):
    # Set the current price to a random price within the last timestep's
    # open price 0 to close price 3
    self.current_step += 1

    current_price = random.uniform(self.stock_df.iloc[-1, 0],
                                   self.stock_df.iloc[-1, 3])
    self.current_price = current_price
    self.total_asset_value = self.cash + self.stock_volume * current_price

    self.volume_to_trade = 0
    action_type = action[0]
    percentage = action[1]
    if action_type <= 1:
      # Trade. If action_type > 1 then hold
      target_stock_value = self.total_asset_value * percentage
      target_stock_volume \
        = target_stock_value // current_price // MIN_TRADE_VOLUME \
        * MIN_TRADE_VOLUME
      volume_to_trade = target_stock_volume - self.stock_volume
      self.volume_to_trade = volume_to_trade
      value_to_trade = volume_to_trade * current_price

      self.stock_volume += volume_to_trade
      self.cash -= value_to_trade

      if volume_to_trade > 0:
        # Buy volume_to_trade stocks
        self.cash -= buy_fee(abs(value_to_trade))
      else:
        self.cash -= sell_fee(abs(value_to_trade))

    self.stock_value = self.stock_volume * current_price
    self.total_asset_value = self.cash + self.stock_value

    if self.total_asset_value > self.max_asset_value:
      self.max_asset_value = self.total_asset_value

  def step(self, action):
    # Execute one time step within the environment
    before_total_asset_value = self.total_asset_value
    self._take_action(action)
    after_total_asset_value = self.total_asset_value
    after_total_asset_value *= (1 - 0.03 / 251)

    reward = after_total_asset_value / before_total_asset_value - 1
    done = ((self.total_asset_value / INIT_CASH) <= 0.5) or self.done

    # When used with Vectorized Env, env will be
    # automatically reset when done
    # if done:
    #   obs = self.reset()
    # else:
    #   obs = self._next_observation()
    obs = np.array([])
    if not done:
      obs = self._next_observation()

    return obs, reward, done, {
        'stock_name': self.stock_name,
        'timestamp': self.stock_df.index,
        'total_asset_value': self.total_asset_value,
        'stock_volume': self.stock_volume,
        'stock_value': self.stock_value,
        'current_price': self.current_price,
        'max_asset_value': self.max_asset_value,
        'volume_to_trade': self.volume_to_trade
    }

  def reset(self):
    # Reset the state of the environment to an initial state
    self.total_asset_value = INIT_CASH
    self.cash = INIT_CASH
    self.stock_volume = 0
    self.stock_value = 0
    self.current_price = 0
    self.volume_to_trade = 0
    self.max_asset_value = 0

    self.current_step = 0

    self.reader = CsiReader(self.seq_len)

    return self._next_observation()

  def render(self, mode=''):
    # Render the environment to the screen
    profit = self.total_asset_value - INIT_CASH

    print(f'Step: {self.current_step}')
    print(f'Balance: {self.balance}')
    print(f'Shares held: {self.stock_volume}')
    print(
        f'Total Asset Value: {self.total_asset_value} (Max Value: {self.max_asset_value})'
    )
    print(f'Profit: {profit}')


class CsiReader:

  def __init__(self, seq_len=1):
    '''
    seq_len: how many timesteps to get at each call
    '''
    self.file_idx = 0
    self.row_idx = 0
    self.seq_len = seq_len
    self.dataset_epoch = 0
    self.stock_feat_dim = stock_feat_dim

    wcsi_df = pd.read_csv(
        wcsi_path,
        dtype={
            'date': np.str,
            'con_code': np.str,
            'weight': np.float,
            'stock_code': np.str,
            'mkt': np.str,
            'filename': np.str
        })

    self.file_paths_dir_stock_list = [(d + i.filename, i.filename.split('.')[0])
                                      for d in csi_dir
                                      for i in wcsi_df.itertuples()]
    shuffle(self.file_paths_dir_stock_list)
    self.files_no = len(self.file_paths_dir_stock_list)

    self.get_new_df_norm_df()

  def get_next_step(self):
    self.is_next_new_stock = False

    end_row_idx = self.row_idx + self.seq_len
    df = self.df.iloc[self.row_idx:end_row_idx, :]
    norm_nd = self.norm_nd[self.row_idx:end_row_idx, :]

    self.row_idx = end_row_idx

    if end_row_idx >= self.current_file_rows:
      self.get_new_df_norm_df()
      end_row_idx = self.row_idx + self.seq_len

    return df, norm_nd, self.is_next_new_stock, self.df_stock

  def get_new_df_norm_df(self):
    while True:  # in case non file
      try:
        if self.file_idx == self.files_no:
          self.file_idx = 0
          self.dataset_epoch += 1

        fp_list = self.file_paths_dir_stock_list[self.file_idx]
        fp = fp_list[0]
        df = pd.read_csv(fp, header=None, index_col=None)
        self.file_idx += 1
        self.df_file_dir = fp_list[0]
        self.df_stock = fp_list[1]

        df.columns = ['date', 'time', 'o', 'h', 'l', 'c', 'v', 'a']
        df.index = pd.to_datetime(
            df['date'] + ' ' + df['time'], format="%Y%m%d %H:%M")
        self.df = df.iloc[:, 2:]
        self.norm_nd = self.normalize_df(self.df)
        self.current_file_rows = df.shape[0]

        self.is_next_new_stock = True
        self.row_idx = 0
        break
      except FileNotFoundError as e:
        self.file_idx += 1

  def normalize_df(self, x):
    # for o,l,h,c use the same scaler
    price_arr = x.iloc[:, :4].values.reshape(-1, 1)
    price_scaler = MinMaxScaler((-1, 1))
    price_scaler.fit(price_arr)
    norm_df = price_scaler.transform(price_arr).reshape(-1, 4)

    # for v,a use the same scaler
    va_scaler = MinMaxScaler((-1, 1))
    va_scaler.fit(x.iloc[:, -2:])
    va_df = va_scaler.transform(x.iloc[:, -2:])

    norm_nd = np.concatenate([norm_df, va_df], axis=1)
    return norm_nd


if __name__ == "__main__":
  reader = CsiReader()
  while True:
    df, norm_nd, is_next_new_stock, stock_name = reader.get_next_step()
    if is_next_new_stock:
      print(df)
      print(norm_nd)
      print(reader.df_file_dir)
      print(reader.df_stock)
    if reader.dataset_epoch == 2:
      break
