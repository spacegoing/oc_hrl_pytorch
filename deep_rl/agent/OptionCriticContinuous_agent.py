#######################################################################
# Copyright (C) 2017 Shangtong Zhang(zhangshangtong.cpp@gmail.com)    #
# Permission given to modify the code as long as you keep this        #
# declaration at the top                                              #
#######################################################################

from ..network import *
from ..component import *
from .BaseAgent import *


class OptionCriticContinuousAgent(BaseAgent):

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
    with torch.no_grad():
      q_option = prediction['q']
      pi_option = torch.zeros_like(q_option).add(epsilon / q_option.size(1))
      greedy_option = q_option.argmax(dim=-1, keepdim=True)
      prob = 1 - epsilon + epsilon / q_option.size(1)
      prob = torch.zeros_like(pi_option).add(prob)
      pi_option.scatter_(1, greedy_option, prob)

      mask = torch.zeros_like(q_option)
      mask[:, prev_option] = 1
      beta = prediction['beta']
      pi_hat_option = (1 - beta) * mask + beta * pi_option

      dist = torch.distributions.Categorical(probs=pi_option)
      options = dist.sample()
      dist = torch.distributions.Categorical(logits=pi_hat_option)
      options_hat = dist.sample()

      options = torch.where(is_intial_states, options, options_hat)
    return options

  def step(self):
    config = self.config
    storage = Storage(config.rollout_length,
                      ['beta', 'o', 'beta_adv', 'prev_o', 'init', 'eps'])

    for _ in range(config.rollout_length):
      prediction = self.network(self.states)
      epsilon = config.random_option_prob(config.num_workers)
      options = self.sample_option(prediction, epsilon, self.prev_options,
                                   self.is_initial_states)
      prediction['pi'] = prediction['pi'][self.worker_index, options]
      # recurr diff: pi are probs (mean), std are trained params
      dist = torch.distributions.Normal(
          loc=prediction['pi'], scale=F.softplus(self.network.std[options]))
      actions = dist.sample()
      # recurr diff: log_pi here is log_prob of action
      prediction['log_pi'] = dist.log_prob(actions).sum(-1).unsqueeze(-1)
      entropy = dist.entropy().sum(-1)

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
          'ent': entropy.unsqueeze(-1),
          'a': actions.unsqueeze(-1),
          'init': self.is_initial_states.unsqueeze(-1).float(),
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
      betas = prediction['beta'][self.worker_index, self.prev_options]
      ret = (1 - betas) * prediction['q'][self.worker_index, self.prev_options] + \
            betas * torch.max(prediction['q'], dim=-1)[0]
      ret = ret.unsqueeze(-1)

    for i in reversed(range(config.rollout_length)):
      ret = storage.r[i] + config.discount * storage.m[i] * ret
      adv = ret - storage.q[i].gather(1, storage.o[i])
      storage.ret[i] = ret
      storage.adv[i] = adv

      v = storage.q[i].max(
          dim=-1, keepdim=True)[0] * (1 - storage.eps[i]) + storage.q[i].mean(
              -1).unsqueeze(-1) * storage.eps[i]
      q = storage.q[i].gather(1, storage.prev_o[i])
      storage.beta_adv[i] = q - v + config.termination_regularizer

    q, beta, log_pi, ret, adv, beta_adv, ent, option, action, initial_states, prev_o = \
        storage.cat(['q', 'beta', 'log_pi', 'ret', 'adv', 'beta_adv', 'ent', 'o', 'a', 'init', 'prev_o'])

    if config.log_interval and not self.total_steps % config.log_interval:
      self.logger.add_scalar('input_info/return',
                             ret.mean().detach(), self.total_steps)
      self.logger.add_scalar('input_info/adv_o',
                             adv.mean().detach(), self.total_steps)
      self.logger.add_scalar('input_info/adv_beta',
                             beta_adv.mean().detach(), self.total_steps)

    q_loss = (q.gather(1, option) - ret.detach()).pow(2).mul(0.5).mean()
    # recurr diff: log_pi here is log_prob of action, no need to gather action
    pi_loss = -(log_pi * adv.detach()) - config.entropy_weight * ent
    pi_loss = pi_loss.mean()
    beta_loss = (beta.gather(1, prev_o) * beta_adv.detach() *
                 (1 - initial_states)).mean()
    total_loss = pi_loss + q_loss + beta_loss

    if config.log_interval and not self.total_steps % config.log_interval:
      self.logger.add_scalar('loss/total_loss', total_loss.detach(),
                             self.total_steps)
      self.logger.add_scalar('loss/entropy_loss',
                             ent.mean().detach(), self.total_steps)
      self.logger.add_scalar('loss/option_loss', pi_loss.detach(),
                             self.total_steps)
      self.logger.add_scalar('loss/q_omega_loss', q_loss.detach(),
                             self.total_steps)
      self.logger.add_scalar('loss/beta_loss', beta_loss.detach(),
                             self.total_steps)

    self.optimizer.zero_grad()
    total_loss.backward()
    nn.utils.clip_grad_norm_(self.network.parameters(), config.gradient_clip)
    self.optimizer.step()
