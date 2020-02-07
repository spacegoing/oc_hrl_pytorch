#######################################################################
# Copyright (C) 2017 Shangtong Zhang(zhangshangtong.cpp@gmail.com)    #
# Permission given to modify the code as long as you keep this        #
# declaration at the top                                              #
#######################################################################

from ..network import *
from ..component import *
from .BaseAgent import *
from random import shuffle


class PPOAgent(BaseAgent):

  def __init__(self, config):
    BaseAgent.__init__(self, config)
    self.config = config
    self.task = config.task_fn()
    self.network = config.network_fn()
    self.opt = config.optimizer_fn(self.network.parameters())
    self.total_steps = 0
    self.states = self.task.reset()
    self.states = config.state_normalizer(self.states)

  def rollout(self, storage):
    config = self.config
    states = self.states
    is_recur = self.network.is_recur
    # masks: [timestep, batchsize]. 0 for done step, 1 for continue step
    masks = np.ones(config.num_workers)

    with torch.no_grad():
      if is_recur:
        input_lstm_states = self.network.get_init_lstm_states(
            config.num_workers)
      for _ in range(config.rollout_length):
        if is_recur:
          # obs [num_workers, feat_dim] during rollout, only 1 time step
          # extend to [timesteps, num_workers, feat_dim]
          seq_states = states[np.newaxis, ...]
          masks = masks[np.newaxis, ...]
          prediction = self.network(seq_states, input_lstm_states, masks)
          input_lstm_states = prediction['final_lstm_states']
        else:
          prediction = self.network(states)

        next_states, rewards, terminals, info = self.task.step(
            to_np(prediction['a']))
        self.record_online_return(info)

        rewards = config.reward_normalizer(rewards)
        next_states = config.state_normalizer(next_states)
        masks = tensor(1 - terminals)

        storage.add(prediction)
        storage.add({
            'r': tensor(rewards).unsqueeze(-1),
            'm': masks.unsqueeze(-1),
            'next_s': tensor(next_states),
            's': tensor(states)
        })

        states = next_states
        self.total_steps += config.num_workers

      self.states = states

  def comp_adv(self, storage):
    config = self.config
    states = self.states
    is_recur = self.network.is_recur

    with torch.no_grad():
      if is_recur:
        # obs [num_workers, feat_dim] during rollout, only 1 time step
        # extend to [timesteps, num_workers, feat_dim]
        states = states[np.newaxis, ...]
        masks = np.ones([1, config.num_workers])
        prediction = self.network(states, storage.final_lstm_states[-1], masks)
      else:
        prediction = self.network(states)
      storage.add(prediction)
      storage.placeholder()

      advantages = tensor(np.zeros((config.num_workers, 1)))
      returns = prediction['v']
      for i in reversed(range(config.rollout_length)):
        returns = storage.r[i] + config.discount * storage.m[i] * returns
        if not config.use_gae:
          advantages = returns - storage.v[i]
        else:
          td_error = storage.r[i] + config.discount * storage.m[i] * storage.v[
              i + 1] - storage.v[i]
          advantages = advantages * config.gae_tau * config.discount * storage.m[
              i] + td_error
        storage.adv[i] = advantages
        storage.ret[i] = returns

  def _train_step(self,
                  sampled_states,
                  sampled_actions,
                  sampled_log_probs_old,
                  sampled_returns,
                  sampled_advantages,
                  input_lstm_states=None,
                  sampled_masks=None,
                  num_workers=None):
    config = self.config
    final_lstm_states = None

    if self.network.is_recur:
      sampled_states, sampled_actions = [
          i.reshape([config.rollout_length, num_workers, -1])
          for i in [sampled_states, sampled_actions]
      ]
      sampled_masks = sampled_masks.reshape(
          [config.rollout_length, num_workers])
      prediction = self.network(sampled_states, input_lstm_states,
                                sampled_masks, sampled_actions)
      final_lstm_states = prediction['final_lstm_states']
    else:
      prediction = self.network(sampled_states, sampled_actions)

    ratio = (prediction['log_pi_a'] - sampled_log_probs_old).exp()
    obj = ratio * sampled_advantages
    obj_clipped = ratio.clamp(1.0 - self.config.ppo_ratio_clip, 1.0 +
                              self.config.ppo_ratio_clip) * sampled_advantages
    policy_loss = -torch.min(obj, obj_clipped).mean(
    ) - config.entropy_weight * prediction['ent'].mean()

    value_loss = 0.5 * (sampled_returns - prediction['v']).pow(2).mean()

    self.opt.zero_grad()
    (policy_loss + value_loss).backward()
    nn.utils.clip_grad_norm_(self.network.parameters(), config.gradient_clip)
    self.opt.step()

    return final_lstm_states

  def step(self):
    config = self.config
    storage = Storage(config.rollout_length)
    is_recur = self.network.is_recur

    self.rollout(storage)
    self.comp_adv(storage)

    states, actions, log_probs_old, returns, advantages, masks = storage.cat(
        ['s', 'a', 'log_pi_a', 'ret', 'adv', 'm'])
    advantages = (advantages - advantages.mean()) / advantages.std()

    for _ in range(config.optimization_epochs):
      if is_recur:
        # [batch_size, feat_dim] -> [seq_len, batch_size, feat_dim]
        states, actions, log_probs_old, returns, advantages, masks = [
            i.reshape([config.rollout_length, config.num_workers, -1]) for i in
            [states, actions, log_probs_old, returns, advantages, masks]
        ]
        worker_idxs = list(range(self.config.num_workers))
        shuffle(worker_idxs)
        for start_worker_idx in range(0, self.config.num_workers, 3):
          batch_indices = worker_idxs[start_worker_idx:start_worker_idx + 3]
          num_workers = len(batch_indices)
          input_lstm_states = self.network.get_init_lstm_states(num_workers)
          sampled_states = states[:, batch_indices, :]
          sampled_actions = actions[:, batch_indices, :]
          sampled_log_probs_old = log_probs_old[:, batch_indices, :]
          sampled_returns = returns[:, batch_indices, :]
          sampled_advantages = advantages[:, batch_indices, :]
          sampled_masks = masks[:, batch_indices, :]
          sampled_states, sampled_actions, sampled_log_probs_old, \
            sampled_returns, sampled_advantages, sampled_masks = [
              i.reshape([config.rollout_length * num_workers, -1]) for i in [
                  sampled_states, sampled_actions, sampled_log_probs_old,
                  sampled_returns, sampled_advantages, sampled_masks
              ]
          ]
          self._train_step(sampled_states, sampled_actions,
                           sampled_log_probs_old, sampled_returns,
                           sampled_advantages, input_lstm_states, sampled_masks,
                           num_workers)

      else:
        sampler = random_sample(
            np.arange(states.size(0)), config.mini_batch_size)
        for batch_indices in sampler:
          batch_indices = tensor(batch_indices).long()
          sampled_states = states[batch_indices]
          sampled_actions = actions[batch_indices]
          sampled_log_probs_old = log_probs_old[batch_indices]
          sampled_returns = returns[batch_indices]
          sampled_advantages = advantages[batch_indices]

          self._train_step(sampled_states, sampled_actions,
                           sampled_log_probs_old, sampled_returns,
                           sampled_advantages)
