# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import gym
from gym import spaces


class CSI300(gym.Env):
  # Because of google colab, we cannot implement the GUI ('human' render mode)
  metadata = {'render.modes': ['console']}
  # Define constants for clearer code
  LEFT = 0
  RIGHT = 1

  def __init__(self, grid_size=10):
    super().__init__()

    # Size of the 1D-grid
    self.grid_size = grid_size
    # Initialize the agent at the right of the grid
    self.agent_pos = grid_size - 1

    # Define action and observation space
    # They must be gym.spaces objects
    # Example when using discrete actions, we have two: left and right
    n_actions = 2
    self.action_space = spaces.Discrete(n_actions)
    # The observation will be the coordinate of the agent
    # this can be described both by Discrete and Box space
    self.observation_space = spaces.Box(
        low=0, high=self.grid_size, shape=(1,), dtype=np.float32)

  def reset(self):
    """
    Important: the observation must be a numpy array
    :return: (np.array)
    """
    # Initialize the agent at the right of the grid
    self.agent_pos = self.grid_size - 1
    # here we convert to float32 to make it more general (in case we want to use continuous actions)
    return np.array([self.agent_pos]).astype(np.float32)

  def step(self, action):
    if action == self.LEFT:
      self.agent_pos -= 1
    elif action == self.RIGHT:
      self.agent_pos += 1
    else:
      raise ValueError(
          "Received invalid action={} which is not part of the action space".
          format(action))

    # Account for the boundaries of the grid
    self.agent_pos = np.clip(self.agent_pos, 0, self.grid_size)

    # Are we at the left of the grid?
    done = bool(self.agent_pos == 0)

    # Null reward everywhere except when reaching the goal (left of the grid)
    reward = 1 if self.agent_pos == 0 else 0

    # Optionally we can pass additional info, we are not using that for now
    info = {}

    return np.array([self.agent_pos]).astype(np.float32), reward, done, info

  def render(self, mode='console'):
    if mode != 'console':
      raise NotImplementedError()
    # agent is represented as a cross, rest as a dot
    print("." * self.agent_pos, end="")
    print("x", end="")
    print("." * (self.grid_size - self.agent_pos))

  def close(self):
    pass
