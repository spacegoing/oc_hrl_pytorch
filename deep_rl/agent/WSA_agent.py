#######################################################################
# Copyright (C) 2017 Shangtong Zhang(zhangshangtong.cpp@gmail.com)    #
# Permission given to modify the code as long as you keep this        #
# declaration at the top                                              #
#######################################################################

from ..network import *
from ..component import *
from .BaseAgent import *
from skimage import color
import numpy as np
import pickle
from random import shuffle
from pymongo import MongoClient
import traceback

client = MongoClient('mongodb://localhost:27017')
db = client['sa']
debug_flag = True


def generate_lag_seq_mat(single_step_mat, lag=1):
  '''
  Parameters:
    single_step_mat: storage.prev_o [num_workers, total_timesteps]
    lag: interger [t-lag, ... , t-1]

  Return:
    lag_mat: [num_workers, total_timesteps, lag]
             for empty timesteps, time step 1 does not have
             value 1-1 ... 1-k, all empty timesteps' value
             are -1
  '''
  mat = single_step_mat
  lag_mat = tensor(np.zeros([mat.shape[0], mat.shape[1], lag]))
  lag_mat[...] = -1

  for b in range(mat.shape[0]):
    # b batch index
    for t in range(mat.shape[1]):
      start = t + 1 - lag if (t + 1) > lag else 0
      lag_mat[b, t, :t + 1] = mat[b][start:t + 1]
  return lag_mat


class WsaAgent(BaseAgent):

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
    self.initial_state_flags = tensor(np.ones((config.num_workers))).bool()

    self.count = 0
    self.exp_col = db[config.log_file_apdx]
    self.error_col = db[config.log_file_apdx + '_error']

    self.env = self.task.env.envs[0].env
    self.task_switch_flag = False

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
      o_ret, o_adv, a_ret, a_adv, prev_options, init = storage.cat(
        ['s', 'at', 'pat_log_prob', 'ot', 'po_t_log', \
         'o_ret', 'o_adv', 'a_ret', 'a_adv', 'prev_o', 'init'])
    a_adv = (a_adv - a_adv.mean()) / a_adv.std()
    o_adv = (o_adv - o_adv.mean()) / o_adv.std()

    def embed_cosine_loss(wt, eps=1e-8):
      """
      added eps for numerical stability
      """
      w_n = wt.norm(dim=1).unsqueeze(-1)
      w_norm = wt / torch.max(w_n, eps * torch.ones_like(w_n))
      sim_mt = torch.mm(w_norm, w_norm.transpose(0, 1))
      low_diagonal = torch.tril(sim_mt, diagonal=-1)
      num = (low_diagonal != 0).sum()
      cosine_loss = low_diagonal.sum() / num
      return cosine_loss

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
                     sampled_a_adv, sampled_a_ret, misc):
      # mean/std: [num_workers, action_dim]
      pat_mean = prediction['pat_mean']
      pat_std = prediction['pat_std']
      pat_dist = torch.distributions.Normal(pat_mean, pat_std)
      # pat_log_prob_new: [num_workers, 1]
      pat_log_prob_new = pat_dist.log_prob(sampled_at_old).sum(-1).unsqueeze(-1)

      pat_loss = ppo_loss(pat_log_prob_new, sampled_pat_log_prob_old,
                          sampled_a_adv, self.config.ppo_ratio_clip_action)

      pat_ent = pat_dist.entropy().sum(-1).mean()
      q_loss = (prediction['q_ot_st'] - sampled_a_ret).pow(2).mul(0.5).mean()
      return pat_loss + q_loss - config.a_entropy_weight * pat_ent

    def learn_option(prediction, sampled_ot_old, sampled_po_t_log_prob_old,
                     sampled_o_adv, sampled_o_ret, misc):
      po_t_ent = -(prediction['po_t_log'] * prediction['po_t']).sum(-1).mean()
      pot_log_prob_new = prediction['po_t_log'].gather(1, sampled_ot_old)
      pot_log_prob_old = sampled_po_t_log_prob_old.gather(1, sampled_ot_old)
      option_clip_ratio = self._option_clip_schedular()
      pot_loss = ppo_loss(pot_log_prob_new, pot_log_prob_old, sampled_o_adv,
                          option_clip_ratio)
      pot_loss = pot_loss - config.o_entropy_weight * po_t_ent

      q_loss = (prediction['v_st'] - sampled_o_ret).pow(2).mul(0.5).mean()
      cosine_loss = embed_cosine_loss(misc['wt'], eps=1e-8)
      return pot_loss + q_loss + config.cos_w * cosine_loss

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
          if debug_flag == True:
            import ipdb
            ipdb.set_trace(context=7)
          batch_indices = tensor(batch_indices).long()
          sampled_states = states[batch_indices]
          sampled_prev_options = prev_options[batch_indices]
          sampled_initial_flags = init[batch_indices]
          prediction = self.network(sampled_states, sampled_prev_options,
                                    sampled_initial_flags,
                                    self.task_switch_flag)

          misc = dict()
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
            misc = {'wt': prediction['wt']}

          loss = learn_fn(
              prediction,
              sampled_action_old,
              sampled_log_prob_old,
              sampled_adv,
              sampled_ret,
              misc,
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
        if debug_flag == True:
          import ipdb
          ipdb.set_trace(context=7)
        prediction = self.network(states, self.prev_options,
                                  self.initial_state_flags,
                                  self.task_switch_flag)

        # mean/std: [num_workers, action_dim]
        pat_mean = prediction['pat_mean']
        pat_std = prediction['pat_std']
        pat_dist = torch.distributions.Normal(pat_mean, pat_std)
        # actions: [num_workers, action_dim]
        at = pat_dist.sample()
        # pi_at: [num_workers, 1]
        pat_log_prob = pat_dist.log_prob(at).sum(-1).unsqueeze(-1)

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
            'init': tensor(terminals).bool().unsqueeze(-1),
            'prev_o': self.prev_options,
            'at': at,
            'pat_log_prob': pat_log_prob,
        })

        self.initial_state_flags = tensor(terminals).bool()
        self.prev_options = prediction['ot']
        states = next_states
        self.total_steps += config.num_workers

        if config.log_analyze_stat:
          try:
            store_dict = {
                's': states,
                'r': np.expand_dims(rewards, axis=-1),
                'm': np.expand_dims(1 - terminals, axis=-1),
                'init': tensor(terminals).bool().unsqueeze(-1),
                'at': to_np(at),
                'ot': to_np(prediction['ot']),
                'po_t': to_np(prediction['po_t']),
                'q_o_st': to_np(prediction['q_o_st']),
                'sim_state': self.env.sim.get_state().flatten(),
            }
            mongo_dict = {k: store_dict[k].tolist() for k in store_dict}
            mongo_dict['step'] = self.total_steps
            self.exp_col.insert_one(mongo_dict)
          except Exception as e:
            self.error_col.insert_one({
                'step': self.total_steps,
                'error': str(e),
                'tradeback': str(traceback.format_exc())
            })

      self.states = states
      # add T+1 step
      prediction = self.network(states, self.prev_options,
                                self.initial_state_flags, self.task_switch_flag)
      storage.add(prediction)
      # padding storage
      storage.placeholder()

  def step(self):
    config = self.config
    storage = Storage(config.rollout_length,
                      ['a_adv', 'o_adv', 'a_ret', 'o_ret'])
    states = self.states
    if self.task_ind > 0:
      self.task_switch_flag = True
      self.network.embed_option.requires_grad_(False)
    self.rollout(storage, config, states)
    self.compute_adv(storage)
    self.learn(storage)