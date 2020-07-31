#######################################################################
# Copyright (C) 2017 Shangtong Zhang(zhangshangtong.cpp@gmail.com)    #
# Permission given to modify the code as long as you keep this        #
# declaration at the top                                              #
#######################################################################

from ..network import *
from ..component import *
from .BaseAgent import *
from skimage import color
import pickle


class DoeAgent(BaseAgent):

  def __init__(self, config):
    BaseAgent.__init__(self, config)
    self.config = config
    self.task = config.task_fn()
    self.network = config.network_fn()
    self.opt = config.optimizer_fn(self.network.parameters())
    self.total_steps = 0

    self.worker_index = tensor(np.arange(config.num_workers)).long()
    self.states = self.task.reset()
    self.states = config.state_normalizer(self.states)
    self.is_initial_states = tensor(np.ones((config.num_workers))).byte()
    self.prev_options = tensor(np.zeros(config.num_workers)).long()

    self.count = 0

    self.all_options = []
    self.logallsteps_storage = []

  def _option_clip_schedular(self):
    return self.config.ppo_ratio_clip_option_max - (
        self.config.ppo_ratio_clip_option_max -
        self.config.ppo_ratio_clip_option_min) * self.total_steps

  def compute_adv(self, storage):
    config = self.config

    q_ot_st = storage.q_ot_st
    adv = storage.adv
    all_ret = storage.ret

    with torch.no_grad():
      # ret: [num_workers, 1]
      ret = q_ot_st[-1]
      # advantages: [num_workers, 1]
      advantages = tensor(np.zeros((config.num_workers, 1)))
      for i in reversed(range(config.rollout_length)):
        # m: [num_workers, 1]
        ret = storage.r[i] + config.discount * storage.m[i] * ret
        if not config.use_gae:
          advantages = ret - q_ot_st[i]
        else:
          # td_error: [num_workers, 1]
          td_error = storage.r[i] +\
            config.discount * storage.m[i] * q_ot_st[i+1] - q_ot_st[i]
          advantages = advantages * config.gae_tau * config.discount *\
            storage.m[i] + td_error
        adv[i] = advantages
        all_ret[i] = ret

  def learn(self, storage):
    config = self.config

    states, at_old, pat_log_prob_old, ot_old, po_t_log_prob_old, \
      returns, advantages, inits, prev_options = storage.cat(
        ['s', 'at', 'pat_log_prob', 'ot', 'po_t_log', \
          'ret', 'adv', 'init','prev_o'])
    advantages = (advantages - advantages.mean()) / advantages.std()

    for _ in range(config.optimization_epochs):
      sampler = random_sample(np.arange(states.size(0)), config.mini_batch_size)
      for batch_indices in sampler:
        '''
        batch_size=mini_batch_size

        batch_indices: [batch_size]
        sampled_states: [batch_size, state_dim]
        sampled_actions: [batch_size, act_dim]
        sampled_options: [batch_size, 1]
        sampled_log_pi_bar_old: [batch_size, 1]
        sampled_returns: [batch_size, 1]
        sampled_advantages: [batch_size, 1]
        sampled_inits: [batch_size, 1]
        sampled_prev_options: [batch_size, 1]
        '''
        batch_indices = tensor(batch_indices).long()
        batch_all_index = range_tensor(len(batch_indices))
        sampled_states = states[batch_indices]
        sampled_at_old = at_old[batch_indices]
        sampled_pat_log_prob_old = pat_log_prob_old[batch_indices]
        sampled_ot_old = ot_old[batch_indices]
        sampled_po_t_log_prob_old = po_t_log_prob_old[batch_indices]
        sampled_returns = returns[batch_indices]
        sampled_advantages = advantages[batch_indices]
        sampled_inits = inits[batch_indices]
        sampled_prev_options = prev_options[batch_indices]

        prediction = self.network(sampled_states, sampled_prev_options)

        # mean/std: [num_workers, action_dim]
        pat_mean = prediction['pat_mean']
        pat_std = prediction['pat_std']
        pat_dist = torch.distributions.Normal(pat_mean, pat_std)
        # pat_new: [num_workers, 1]
        pat_new = pat_dist.log_prob(sampled_at_old).sum(-1).exp().unsqueeze(-1)
        pat_log_prob_new = pat_new.add(1e-5).log()

        pat_ratio = (pat_log_prob_new - sampled_pat_log_prob_old).exp()
        pat_obj = pat_ratio * sampled_advantages
        pat_obj_clipped = pat_ratio.clamp(
            1.0 - self.config.ppo_ratio_clip_action,
            1.0 + self.config.ppo_ratio_clip_action) * sampled_advantages
        pat_loss = -torch.min(pat_obj, pat_obj_clipped).mean()

        pot_log_prob_new = prediction['po_t_log'][batch_all_index,
                                                  sampled_ot_old.squeeze(-1)]
        pot_log_prob_old = sampled_po_t_log_prob_old[batch_all_index,
                                                     sampled_ot_old.squeeze(-1)]
        pot_ratio = (pot_log_prob_new - pot_log_prob_old).exp()
        pot_obj = pot_ratio * sampled_advantages
        option_clip_ratio = self._option_clip_schedular()
        pot_obj_clipped = pot_ratio.clamp(
            1.0 - option_clip_ratio,
            1.0 + option_clip_ratio) * sampled_advantages
        pot_loss = -torch.min(pot_obj, pot_obj_clipped).mean()
        po_t_ent = -(prediction['po_t_log'] * prediction['po_t']).sum(-1).mean()
        pot_loss = pot_loss - config.entropy_weight * po_t_ent

        q_loss = (prediction['q_o_st'].gather(1, sampled_ot_old) -
                  sampled_returns).pow(2).mul(0.5).mean()

        self.opt.zero_grad()
        (pat_loss + q_loss + pot_loss).backward()
        nn.utils.clip_grad_norm_(self.network.parameters(),
                                 config.gradient_clip)
        self.opt.step()

  def rollout(self, storage, config, states):
    '''
    Naming Conventions:
    if o does not follow timestamp t, it means for all options:
      q_o_st: [num_workers, num_options] $Q_o_t(O,S_t)$
      po_t/po_t_log: [num_workers, num_options] $P(O|S_t,o_{t-1};w_t)$

    if ot, it means for O=ot:
      q_ot_st: [num_workers, 1] $Q_o_t(O=ot, S_t)$
      pot/pot_log: [num_workers, 1] $P(O=ot|S_t,o_{t-1};w_t)$
    '''
    with torch.no_grad():
      for _ in range(config.rollout_length):
        prediction = self.network(states, self.prev_options.unsqueeze(1))
        ot = prediction['ot']

        self.logger.add_scalar('o_t', ot, log_level=5)
        self.logger.add_scalar('o_t_0', ot[0], log_level=5)
        self.logger.add_scalar(
            'po_t_ent', prediction['po_t_dist'].entropy(), log_level=5)
        self.logger.add_scalar(
            'pot_log_prob', prediction['po_t_dist'].log_prob(ot), log_level=5)

        # mean/std: [num_workers, action_dim]
        pat_mean = prediction['pat_mean']
        pat_std = prediction['pat_std']
        pat_dist = torch.distributions.Normal(pat_mean, pat_std)
        # actions: [num_workers, action_dim]
        at = pat_dist.sample()
        # pi_at: [num_workers, 1]
        pat = pat_dist.log_prob(at).sum(-1).exp().unsqueeze(-1)

        # next_states: tuple([state_dim] * num_workers)
        # terminals(bool)/rewards: [num_workers]
        # info: dict(['reward_run', 'reward_ctrl', 'episodic_return'] * 3)
        next_states, rewards, terminals, info = self.task.step(to_np(at))
        self.record_online_return(info)
        rewards = config.reward_normalizer(rewards)
        # next_states: -> [num_workers, state_dim]
        next_states = config.state_normalizer(next_states)

        # remove ot from prediction, added below due to requiring unsqueeze
        prediction.pop('ot')
        storage.add(prediction)
        '''
          r: [num_workers, 1]
          m: [num_workers, 1]
            0 for terminated states; 1 for continue
          at: [num_workers, act_dim]
          ot: [num_workers, 1]
          prev_o: [num_workers, 1]
          s: [num_workers, state_dim]
          init: [num_workers, 1]
          q_ot_st: [num_workers, 1]
          log_pi_bar: [num_workers, 1]
        '''
        storage.add({
            'r': tensor(rewards).unsqueeze(-1),
            'm': tensor(1 - terminals).unsqueeze(-1),
            'at': at,
            'ot': ot.unsqueeze(-1),
            'prev_o': self.prev_options.unsqueeze(-1),
            's': tensor(states),
            'init': self.is_initial_states.unsqueeze(-1),
            'q_ot_st': prediction['q_o_st'][self.worker_index, \
                                             ot].unsqueeze(-1),
            'pat_log_prob': pat.add(1e-5).log(),
        })

        # self.is_initial_states: [num_workers, 1]
        #    0 for continue; 1 for terminated
        self.is_initial_states = tensor(terminals).byte()
        self.prev_options = ot

        states = next_states
        self.total_steps += config.num_workers

        if config.log_analyze_stat:
          # log analyze stats
          self.logallsteps_storage.append({
              's': states,
              'r': np.expand_dims(rewards, axis=-1),
              'm': np.expand_dims(1 - terminals, axis=-1),
              'at': to_np(at),
              'ot': to_np(ot.unsqueeze(-1)),
              'pot_ent': to_np(prediction['po_t_dist'].entropy().unsqueeze(-1)),
              'q_ot_st': to_np(prediction['q_o_st']),
          })

      self.states = states
      prediction = self.network(states, self.prev_options.unsqueeze(-1))
      ot = prediction['ot']

      prediction.pop('ot')
      storage.add(prediction)
      # v: [num_workers, 1]
      storage.add({
          'q_ot_st': prediction['q_o_st'][self.worker_index, ot].unsqueeze(-1)
      })
      storage.placeholder()

      if config.log_analyze_stat and self.total_steps % (config.max_steps //
                                                         10):
        with open('./analyze/%s' % (config.log_file_apdx), 'wb') as f:
          pickle.dump(self.logallsteps_storage, f)

  def step(self):
    config = self.config
    storage = Storage(config.rollout_length)
    states = self.states
    self.rollout(storage, config, states)
    self.compute_adv(storage)
    self.learn(storage)
