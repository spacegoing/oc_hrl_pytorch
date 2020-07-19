#######################################################################
# Copyright (C) 2017 Shangtong Zhang(zhangshangtong.cpp@gmail.com)    #
# Permission given to modify the code as long as you keep this        #
# declaration at the top                                              #
#######################################################################

from ..network import *
from ..component import *
from .BaseAgent import *


class OCAgent(BaseAgent):

  def __init__(self, config):
    BaseAgent.__init__(self, config)
    self.config = config
    self.task = config.task_fn()
    self.network = config.network_fn()
    self.target_network = config.network_fn()
    self.optimizer = config.optimizer_fn(self.network.parameters())
    self.target_network.load_state_dict(self.network.state_dict())

    self.total_steps = 0
    self.worker_index = tensor(np.arange(config.num_workers)).long()

    self.states = self.config.state_normalizer(self.task.reset())
    self.is_initial_states = tensor(np.ones((config.num_workers))).byte()
    self.prev_options = self.is_initial_states.clone().long()

  def sample_option(self, prediction, epsilon, prev_option, is_intial_states):
    '''
    epsilon-greedy policy over option
      epsilon = 0.1
      single_option_eps = epsilon/num_options(4) = 0.025
      > prob = 1 - (num_options - 1) * single_option_epsilon = 0.925

      Then, assign prob to greedy option of each worker
      > pi_option: 3*0.025 + 0.925 = 1
      tensor([[0.0250, 0.9250, 0.0250, 0.0250],
              [0.0250, 0.0250, 0.0250, 0.9250],
              [0.0250, 0.0250, 0.9250, 0.0250]], device='cuda:0')
    '''
    with torch.no_grad():
      # q_option/pi_option: [num_workers, num_options]
      q_option = prediction['q_o']
      num_workers, num_options = q_option.size()
      single_option_epsilon = epsilon / num_options
      pi_option = torch.zeros_like(q_option).add(single_option_epsilon)
      # greedy_option: [num_workers, 1]
      greedy_option = q_option.argmax(dim=-1)
      prob = 1 - (num_options - 1) * single_option_epsilon
      # pi_option: [num_workers, num_options]
      pi_option[self.worker_index, greedy_option] = prob

      # mask/beta: [num_workers, num_options]
      mask = torch.zeros_like(q_option)
      mask[self.worker_index, prev_option] = 1
      beta = prediction['beta']
      pi_hat_option = (1 - beta) * mask + beta * pi_option

      dist = torch.distributions.Categorical(probs=pi_option)
      options = dist.sample()
      dist = torch.distributions.Categorical(logits=pi_hat_option)
      options_hat = dist.sample()

      # If episode ends and restarted, no beta, use original pi_option
      options = torch.where(is_intial_states, options, options_hat)
    return options

  def step(self):
    config = self.config
    storage = Storage(config.rollout_length,
                      ['beta', 'o', 'beta_adv', 'prev_o', 'init', 'eps'])

    for _ in range(config.rollout_length):
      prediction = self.network(self.states)
      epsilon = config.random_option_prob(config.num_workers)
      # options: [num_workers] values: [0,num_workers-1]
      options = self.sample_option(prediction, epsilon, self.prev_options,
                                   self.is_initial_states)

      # mean/std: [num_workers, action_dim]
      mean = prediction['mean'][self.worker_index, options]
      std = prediction['std'][self.worker_index, options]
      dist = torch.distributions.Normal(mean, std)
      # actions: [num_workers, action_dim]
      actions = dist.sample()
      # log_pi/entropy: [num_workers, 1]
      # dist.log_prob(actions)[num_workers, action_dim]
      # .sum(-1)[num_workers].unsqueeze(-1)[num_workers, 1]
      log_pi = dist.log_prob(actions).sum(-1).unsqueeze(-1)
      entropy = dist.entropy().sum(-1).unsqueeze(-1)

      # next_states: [num_workers, state_dim]
      # rewards/terminals: [num_workers] float/bool
      # info: (halfcheetah)['reward_run', 'reward_ctrl', 'episodic_return']
      next_states, rewards, terminals, info = self.task.step(to_np(actions))
      self.record_online_return(info)
      next_states = config.state_normalizer(next_states)
      rewards = config.reward_normalizer(rewards)
      storage.add(prediction)
      storage.add({
          'r': tensor(rewards).unsqueeze(-1),
          'm': tensor(1 - terminals).unsqueeze(-1),
          'o': options.unsqueeze(-1),
          'prev_o': self.prev_options.unsqueeze(-1),
          'ent': entropy,
          'a': actions.unsqueeze(-1),
          'init': self.is_initial_states.unsqueeze(-1).float(),
          'log_pi': log_pi,
          'eps': epsilon
      })

      self.is_initial_states = tensor(terminals).byte()
      self.prev_options = options
      self.states = next_states

      self.total_steps += config.num_workers
      if self.total_steps // config.num_workers % config.target_network_update_freq == 0:
        self.target_network.load_state_dict(self.network.state_dict())

    with torch.no_grad():
      prediction = self.target_network(self.states)
      storage.placeholder()
      # betas: [num_workers]
      betas = prediction['beta'][self.worker_index, self.prev_options]
      # ret: [num_workers]
      # prediction['q_o'][self.worker_index, self.prev_options]: [num_workers]
      # torch.max(prediction['q_o'], dim=-1).values: [num_workers]
      ret = (1 - betas) * prediction['q_o'][self.worker_index, self.prev_options] + \
            betas * torch.max(prediction['q_o'], dim=-1).values
      # ret: [num_workers, 1]
      ret = ret.unsqueeze(-1)

    for i in reversed(range(config.rollout_length)):
      ret = storage.r[i] + config.discount * storage.m[i] * ret
      # storage.q_o[i].gather(1, storage.o[i]): [num_workers, 1]
      # storage.o[i]: [num_workers, 1], O_t
      adv = ret - storage.q_o[i].gather(1, storage.o[i])
      storage.ret[i] = ret
      storage.adv[i] = adv

      # v: [num_workers, 1]
      # storage.q_o[i].max(dim=-1, keepdim=True).values: [num_workers, 1]
      # storage.q_o[i].mean(-1).unsqueeze(-1): [num_workers, 1]
      v = storage.q_o[i].max(
            dim=-1, keepdim=True).values * (1 - storage.eps[i]) +\
          storage.q_o[i].mean(-1).unsqueeze(-1) * storage.eps[i]
      # q: [num_workers, 1]
      q = storage.q_o[i].gather(1, storage.prev_o[i])
      # beta_adv: [num_workers, 1]
      storage.beta_adv[i] = q - v + config.beta_reg

    q, beta, log_pi, ret, adv, beta_adv, ent, option, action, initial_states, prev_o = \
        storage.cat(['q_o', 'beta', 'log_pi', 'ret', 'adv', 'beta_adv', 'ent', 'o', 'a', 'init', 'prev_o'])

    '''
      q: [num_workers*rollout, num_options]
      beta: [num_workers*rollout, num_options]
      log_pi: [num_workers*rollout, 1]
      ret: [num_workers*rollout, 1]
      adv: [num_workers*rollout, 1]
      beta_adv: [num_workers*rollout, 1]
      ent: [num_workers*rollout, 1]
      option: [num_workers*rollout, 1]
      action: [num_workers*rollout, act_dim, 1]
      initial_states: [num_workers*rollout, 1]
      prev_o: [num_workers*rollout, 1]
    '''
    q_loss = (q.gather(1, option) - ret.detach()).pow(2).mul(0.5).mean()
    pi_loss = -(log_pi * adv.detach()) - config.entropy_weight * ent
    pi_loss = pi_loss.mean()
    beta_loss = (beta.gather(1, prev_o) * beta_adv.detach() *
                 (1 - initial_states)).mean()

    self.optimizer.zero_grad()
    (pi_loss + q_loss + beta_loss).backward()
    nn.utils.clip_grad_norm_(self.network.parameters(), config.gradient_clip)
    self.optimizer.step()
