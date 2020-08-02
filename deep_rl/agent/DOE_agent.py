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
from random import shuffle


class DoeAgent(BaseAgent):

  def __init__(self, config):
    BaseAgent.__init__(self, config)
    self.config = config
    self.task = config.task_fn()
    self.network = config.network_fn()
    self.opt = config.optimizer_fn(self.network.parameters())
    self.total_steps = 0

    self.states = self.task.reset()
    self.states = config.state_normalizer(self.states)
    self.prev_options = tensor(np.zeros([config.num_workers, 1])).long()

    self.count = 0

    self.all_options = []
    self.logallsteps_storage = []

  def _option_clip_schedular(self):
    return self.config.ppo_ratio_clip_option_max - (
        self.config.ppo_ratio_clip_option_max -
        self.config.ppo_ratio_clip_option_min) * self.total_steps

  def compute_adv(self, storage):
    config = self.config

    def ppo_advantages(v_st, adv_list, ret_list):
      '''
      v_st: list[rollout_length+1] each entry: [num_workers, 1]
      adv_list: empty list filled with None len=rollout_length+1
      ret_list: empty list filled with None len=rollout_length+1
      '''
      with torch.no_grad():
        # ret: [num_workers, 1]
        #   (current actual reward R_t) + (estimated all following rewards)
        #   R_t + \sum^N\gamma^n V(S_t+n)
        ret = v_st[-1]
        # adv: [num_workers, 1]
        adv = tensor(np.zeros((config.num_workers, 1)))
        for i in reversed(range(config.rollout_length)):
          # m: [num_workers, 1]
          ret = storage.r[i] + config.discount * storage.m[i] * ret
          if not config.use_gae:
            adv = ret - v_st[i]
          else:
            # td_error: [num_workers, 1]
            a_td_error = storage.r[i] +\
              config.discount * storage.m[i] * v_st[i+1] - v_st[i]
            adv = adv * config.gae_tau * config.discount *\
              storage.m[i] + a_td_error
          adv_list[i] = adv
          ret_list[i] = ret

    v_st = storage.v_st
    o_adv_list = storage.o_adv
    o_ret_list = storage.o_ret
    ppo_advantages(v_st, o_adv_list, o_ret_list)

    q_ot_st = storage.q_ot_st  # Q(O_t,S_t) = marg_a Q(a,O,S)
    a_adv_list = storage.a_adv
    a_ret_list = storage.a_ret
    ppo_advantages(q_ot_st, a_adv_list, a_ret_list)

  def learn(self, storage):
    config = self.config

    states, at_old, pat_log_prob_old, ot_old, po_t_log_prob_old,\
      o_ret, o_adv, a_ret, a_adv, prev_options = storage.cat(
        ['s', 'at', 'pat_log_prob', 'ot', 'po_t_log', \
         'o_ret', 'o_adv', 'a_ret', 'a_adv', 'prev_o'])
    a_adv = (a_adv - a_adv.mean()) / a_adv.std()
    o_adv = (o_adv - o_adv.mean()) / o_adv.std()

    def ppo_loss(p_log_new, p_log_old, adv, clip_rate):
      '''
      p_log_new: [num_workers, 1]
      p_log_old: [num_workers, 1]
      adv: [num_workers, 1]
      '''
      p_ratio = (p_log_new - p_log_old).exp()
      p_obj = p_ratio * adv
      p_obj_clipped = p_ratio.clamp(1.0 - clip_rate, 1.0 + clip_rate) * adv
      p_loss = -torch.min(p_obj, p_obj_clipped).mean()
      return p_loss

    def learn_action(prediction, sampled_at_old, sampled_pat_log_prob_old,
                     sampled_a_adv, sampled_a_ret):
      # mean/std: [num_workers, action_dim]
      pat_mean = prediction['pat_mean']
      pat_std = prediction['pat_std']
      pat_dist = torch.distributions.Normal(pat_mean, pat_std)
      # pat_new: [num_workers, 1]
      pat_new = pat_dist.log_prob(sampled_at_old).sum(-1).exp().unsqueeze(-1)
      pat_log_prob_new = pat_new.add(1e-5).log()

      pat_loss = ppo_loss(pat_log_prob_new, sampled_pat_log_prob_old,
                          sampled_a_adv, self.config.ppo_ratio_clip_action)

      q_loss = (prediction['q_ot_st'] - sampled_a_ret).pow(2).mul(0.5).mean()
      return pat_loss + q_loss

    def learn_option(prediction, sampled_ot_old, sampled_po_t_log_prob_old,
                     sampled_o_adv, sampled_o_ret):
      po_t_ent = -(prediction['po_t_log'] * prediction['po_t']).sum(-1).mean()
      pot_log_prob_new = prediction['po_t_log'].gather(1, sampled_ot_old)
      pot_log_prob_old = sampled_po_t_log_prob_old.gather(1, sampled_ot_old)
      option_clip_ratio = self._option_clip_schedular()
      pot_loss = ppo_loss(pot_log_prob_new, pot_log_prob_old, sampled_o_adv,
                          option_clip_ratio)
      pot_loss = pot_loss - config.entropy_weight * po_t_ent

      q_loss = (prediction['v_st'] - sampled_o_ret).pow(2).mul(0.5).mean()
      return pot_loss + q_loss

    learn_fn_list = [[learn_option, 'o'], [learn_action, 'a']]
    if config.shuffle_train:
      shuffle(learn_fn_list)
    for learn_fn, name in learn_fn_list:
      for _ in range(config.optimization_epochs):
        sampler = random_sample(
            np.arange(states.size(0)), config.mini_batch_size)
        for batch_indices in sampler:
          '''
          batch_size=mini_batch_size
          batch_indices: [batch_size]
          states: [batch_size, state_dim]
          at_old: [batch_size, act_dim]
          pat_log_prob_old: [batch_size, 1]
          ot_old: [batch_size, 1]
          po_t_log_prob_old: [batch_size, num_options]
          o_ret: [batch_size, 1]
          o_adv: [batch_size, 1]
          a_ret: [batch_size, 1]
          a_adv: [batch_size, 1]
          prev_options: [batch_size, 1]
          '''
          batch_indices = tensor(batch_indices).long()
          sampled_states = states[batch_indices]
          sampled_prev_options = prev_options[batch_indices]
          prediction = self.network(sampled_states, sampled_prev_options)

          if name == 'a':
            sampled_action_old = at_old[batch_indices]
            sampled_log_prob_old = pat_log_prob_old[batch_indices]
            sampled_adv = a_adv[batch_indices]
            sampled_ret = a_ret[batch_indices]
          if name == 'o':
            sampled_action_old = ot_old[batch_indices]
            sampled_log_prob_old = po_t_log_prob_old[batch_indices]
            sampled_adv = o_adv[batch_indices]
            sampled_ret = o_ret[batch_indices]

          loss = learn_fn(
              prediction,
              sampled_action_old,
              sampled_log_prob_old,
              sampled_adv,
              sampled_ret,
          )
          self.opt.zero_grad()
          loss.backward()
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
        prediction = self.network(states, self.prev_options)

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

        storage.add(prediction)
        '''
          s: [num_workers, state_dim]
          r: [num_workers, 1]
          m: [num_workers, 1] termination mask
            0 for terminated states; 1 for continue
          prev_o: [num_workers, 1]
          at: [num_workers, act_dim]
          pat_log_prob: [num_workers, 1]
        '''
        storage.add({
            's': tensor(states),
            'r': tensor(rewards).unsqueeze(-1),
            'm': tensor(1 - terminals).unsqueeze(-1),
            'prev_o': self.prev_options,
            'at': at,
            'pat_log_prob': pat.add(1e-5).log(),
        })

        self.prev_options = prediction['ot']
        states = next_states
        self.total_steps += config.num_workers

        if config.log_analyze_stat:
          # log analyze stats
          self.logallsteps_storage.append({
              's': states,
              'r': np.expand_dims(rewards, axis=-1),
              'm': np.expand_dims(1 - terminals, axis=-1),
              'at': to_np(at),
              'ot': to_np(prediction['ot']),
              'pot_ent': to_np(prediction['po_t_dist'].entropy().unsqueeze(-1)),
              'q_o_st': to_np(prediction['q_o_st']),
          })

      self.states = states
      # add T+1 step
      prediction = self.network(states, self.prev_options)
      storage.add(prediction)
      # padding storage
      storage.placeholder()

      if config.log_analyze_stat and self.total_steps % (config.max_steps //
                                                         20):
        try:
          with open('./analyze/%s.pkl' % (config.log_file_apdx), 'rb') as f:
            old_logallsteps_storage = pickle.load(f)
        except FileNotFoundError:
          old_logallsteps_storage = []
        with open('./analyze/%s.pkl' % (config.log_file_apdx), 'wb') as f:
          pickle.dump(old_logallsteps_storage + self.logallsteps_storage, f)
        self.logallsteps_storage = []

  def step(self):
    config = self.config
    storage = Storage(config.rollout_length,
                      ['a_adv', 'o_adv', 'a_ret', 'o_ret'])
    states = self.states
    self.rollout(storage, config, states)
    self.compute_adv(storage)
    self.learn(storage)
