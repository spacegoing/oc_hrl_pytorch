from .replay import *
from .random_process import *
from .envs import Task
from gym.envs.registration import registry, register, make, spec

register(
    id='CSI300-v1',
    entry_point='deep_rl.component.CSIENV.CSIENV:CSI300',
    max_episode_steps=10000,
)
