#######################################################################
# Copyright (C) 2017 Shangtong Zhang(zhangshangtong.cpp@gmail.com)    #
# Permission given to modify the code as long as you keep this        #
# declaration at the top                                              #
#######################################################################
from ..network import *
from ..component import *
from .BaseAgent import *
from skimage import color
from random import shuffle


class PPOCAgent(BaseAgent):

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
    self.masks = tensor(np.ones(config.num_workers))

    self.count = 0

    self.all_options = []

  def compute_pi_hat(self, prediction, prev_option, is_intial_states):
    # sample option

    pi_o = prediction['pi_o']
    mask = torch.zeros_like(pi_o)
    mask[self.worker_index, prev_option] = 1
    beta = prediction['beta']
    pi_hat = (1 - beta) * mask + beta * pi_o
    is_intial_states = is_intial_states.view(-1, 1).expand(-1, pi_o.size(1))
    pi_hat = torch.where(is_intial_states, pi_o, pi_hat)
    return pi_hat

  def compute_pi_bar(self, options, action, mean, std):
    # calculate action

    options = options.unsqueeze(-1).expand(-1, -1, mean.size(-1))
    mean = mean.gather(1, options).squeeze(1)
    std = std.gather(1, options).squeeze(1)
    dist = torch.distributions.Normal(mean, std)
    pi_bar = dist.log_prob(action).sum(-1).exp().unsqueeze(-1)
    return pi_bar

  def compute_adv(self, storage):
    config = self.config

    v = storage.v
    adv = storage.adv
    all_ret = storage.ret

    ret = v[-1]
    advantages = tensor(np.zeros((config.num_workers, 1)))
    for i in reversed(range(config.rollout_length)):
      # todo: not consistent with ppo paper
      ret = storage.r[i] + config.discount * storage.m[i] * ret
      if not config.use_gae:
        advantages = ret - v[i]
      else:
        td_error = storage.r[i] + config.discount * storage.m[i] * v[i +
                                                                     1] - v[i]
        advantages = advantages * config.gae_tau * config.discount * storage.m[
            i] + td_error
      adv[i] = advantages
      all_ret[i] = ret

  def learn(self, storage):
    if self.network.is_recur:
      states, actions, log_pi_bar_old, options, returns, advantages,\
      inits, prev_options, masks = storage.recur_cat(
        ['s', 'a', 'log_pi_bar', 'o', 'ret', 'adv', 'init', 'prev_o', 'm'
         ])
    else:
      states, actions, log_pi_bar_old, options, returns, advantages,\
      inits, prev_options = storage.cat(
        ['s', 'a', 'log_pi_bar', 'o', 'ret', 'adv', 'init', 'prev_o'])
    advantages = (advantages - advantages.mean()) / advantages.std()

    for _ in range(self.config.optimization_epochs):
      if not self.network.is_recur:
        sampler = random_sample(
            np.arange(states.size(0)), self.config.mini_batch_size)
        for batch_indices in sampler:
          batch_indices = tensor(batch_indices).long()
          sampled_states = states[batch_indices]
          sampled_actions = actions[batch_indices]
          sampled_options = options[batch_indices]
          sampled_log_pi_bar_old = log_pi_bar_old[batch_indices]
          sampled_returns = returns[batch_indices]
          sampled_advantages = advantages[batch_indices]
          sampled_inits = inits[batch_indices]
          sampled_prev_options = prev_options[batch_indices]
          self._train_step(sampled_states, sampled_actions,
                           sampled_log_pi_bar_old, sampled_options,
                           sampled_returns, sampled_advantages, sampled_inits,
                           sampled_prev_options)
      else:
        worker_idxs = list(range(self.config.num_workers))
        shuffle(worker_idxs)
        for start_worker_idx in range(0, self.config.num_workers, 3):
          batch_indices = worker_idxs[start_worker_idx:start_worker_idx + 3]
          num_workers = len(batch_indices)
          steps_per_batch = self.config.mini_batch_size // num_workers
          for start_step_idx in range(0, self.config.rollout_length,
                                      steps_per_batch):
            pi_o_final_states, q_o_final_states,\
            all_option_final_states = self.network.get_all_init_states(num_workers)
            sampled_states = states[batch_indices,
                                    start_step_idx:start_step_idx +
                                    steps_per_batch]
            sampled_actions = actions[batch_indices,
                                      start_step_idx:start_step_idx +
                                      steps_per_batch]
            sampled_options = options[batch_indices,
                                      start_step_idx:start_step_idx +
                                      steps_per_batch]
            sampled_log_pi_bar_old = log_pi_bar_old[
                batch_indices, start_step_idx:start_step_idx + steps_per_batch]
            sampled_returns = returns[batch_indices,
                                      start_step_idx:start_step_idx +
                                      steps_per_batch]
            sampled_advantages = advantages[batch_indices,
                                            start_step_idx:start_step_idx +
                                            steps_per_batch]
            sampled_inits = inits[batch_indices, start_step_idx:start_step_idx +
                                  steps_per_batch]
            sampled_prev_options = prev_options[batch_indices,
                                                start_step_idx:start_step_idx +
                                                steps_per_batch]
            sampled_masks = masks[batch_indices, start_step_idx:start_step_idx +
                                  steps_per_batch]
            pi_o_final_states, q_o_final_states,\
            all_option_final_states = self._train_step(sampled_states, sampled_actions,
                             sampled_log_pi_bar_old, sampled_options,
                             sampled_returns, sampled_advantages, sampled_inits,
                             sampled_prev_options, sampled_masks, pi_o_final_states,
                             q_o_final_states, all_option_final_states)

  def _train_step(self,
                  sampled_states,
                  sampled_actions,
                  sampled_log_pi_bar_old,
                  sampled_options,
                  sampled_returns,
                  sampled_advantages,
                  sampled_inits,
                  sampled_prev_options,
                  sampled_masks=None,
                  pi_o_final_states=None,
                  q_o_final_states=None,
                  all_option_final_states=None):
    if self.network.is_recur:
      prediction = self.network(sampled_states, pi_o_final_states,
                                q_o_final_states, all_option_final_states,
                                sampled_prev_options, sampled_masks)
      pi_o_final_states, q_o_final_states,\
      all_option_final_states = prediction['pi_o_final_states'],\
                                prediction['q_o_final_states'],\
                                prediction['all_option_final_states']
      sampled_states, sampled_actions, sampled_log_pi_bar_old, \
      sampled_options, sampled_returns, sampled_advantages, \
      sampled_inits, sampled_prev_options = [
        i.view(-1, i.shape[-1]) for i in [
          sampled_states, sampled_actions, sampled_log_pi_bar_old,
          sampled_options, sampled_returns, sampled_advantages,
          sampled_inits, sampled_prev_options
        ]
      ]
    else:
      prediction = self.network(sampled_states)
    # todo: un exp
    pi_bar = self.compute_pi_bar(sampled_options, sampled_actions,
                                 prediction['mean'], prediction['std'])
    log_pi_bar = pi_bar.add(1e-5).log()
    ratio = (log_pi_bar - sampled_log_pi_bar_old).exp()
    obj = ratio * sampled_advantages
    obj_clipped = ratio.clamp(1.0 - self.config.ppo_ratio_clip, 1.0 +
                              self.config.ppo_ratio_clip) * sampled_advantages
    policy_loss = -torch.min(obj, obj_clipped).mean()

    beta_adv = prediction['q_o'].gather(1, sampled_prev_options) - \
               (prediction['q_o'] * prediction['pi_o']).sum(-1).unsqueeze(-1)
    beta_adv = beta_adv.detach() + self.config.beta_reg
    beta_loss = prediction['beta'].gather(
        1, sampled_prev_options) * (1 - sampled_inits).float() * beta_adv
    beta_loss = beta_loss.mean()

    q_loss = (prediction['q_o'].gather(1, sampled_options) -
              sampled_returns).pow(2).mul(0.5).mean()

    ent = -(prediction['log_pi_o'] * prediction['pi_o']).sum(-1).mean()
    pi_o_loss = -(prediction['log_pi_o'].gather(1, sampled_options) * \
                  sampled_advantages).mean() - self.config.entropy_weight * ent

    self.opt.zero_grad()
    (policy_loss + beta_loss + q_loss + pi_o_loss).backward()
    nn.utils.clip_grad_norm_(self.network.parameters(),
                             self.config.gradient_clip)
    self.opt.step()
    return pi_o_final_states, q_o_final_states, all_option_final_states

  def step(self):
    config = self.config
    storage = Storage(config.rollout_length)
    states = self.states

    with torch.no_grad():
      if self.network.is_recur:
        pi_o_final_states, q_o_final_states, \
        all_option_final_states = self.network.get_all_init_states(config.num_workers)
      for _ in range(config.rollout_length):

        # select option
        if self.network.is_recur:
          prediction = self.network(states, pi_o_final_states, q_o_final_states,
                                    all_option_final_states, self.prev_options,
                                    self.masks)
        else:
          prediction = self.network(states)
        pi_hat = self.compute_pi_hat(prediction, self.prev_options,
                                     self.is_initial_states)
        dist = torch.distributions.Categorical(probs=pi_hat)
        options = dist.sample()

        self.logger.add_scalar(
            'beta',
            prediction['beta'][self.worker_index, self.prev_options],
            log_level=5)
        self.logger.add_scalar('option', options[0], log_level=5)
        self.logger.add_scalar('pi_hat_ent', dist.entropy(), log_level=5)
        self.logger.add_scalar(
            'pi_hat_o', dist.log_prob(options).exp(), log_level=5)

        # sample actions
        mean = prediction['mean'][self.worker_index, options]
        std = prediction['std'][self.worker_index, options]
        dist = torch.distributions.Normal(mean, std)
        actions = dist.sample()

        # todo: un exp in function
        pi_bar = self.compute_pi_bar(
            options.unsqueeze(-1), actions, prediction['mean'],
            prediction['std'])

        next_states, rewards, terminals, info = self.task.step(to_np(actions))
        self.record_online_return(info)
        rewards = config.reward_normalizer(rewards)
        next_states = config.state_normalizer(next_states)

        storage.add(prediction)
        storage.add({
            'r': tensor(rewards).unsqueeze(-1),
            'm': tensor(1 - terminals).unsqueeze(-1),
            'a': actions,
            'o': options.unsqueeze(-1),
            'prev_o': self.prev_options.unsqueeze(-1),
            's': tensor(states),
            'init': self.is_initial_states.unsqueeze(-1),
            'v': prediction['q_o'][self.worker_index, options].unsqueeze(-1),
            'log_pi_bar': pi_bar.add(1e-5).log(),
        })

        self.is_initial_states = tensor(terminals).byte()
        self.prev_options = options
        self.masks = storage.m[-1]

        if self.network.is_recur:
          pi_o_final_states, q_o_final_states,\
          all_option_final_states = prediction['pi_o_final_states'],\
                                    prediction['q_o_final_states'],\
                                    prediction['all_option_final_states']

        states = next_states
        self.total_steps += config.num_workers

      # select next option
      self.states = states
      if self.network.is_recur:
        prediction = self.network(states, prediction['pi_o_final_states'],
                                  prediction['q_o_final_states'],
                                  prediction['all_option_final_states'],
                                  self.prev_options, self.masks)
      else:
        prediction = self.network(states)
      pi_hat = self.compute_pi_hat(prediction, self.prev_options,
                                   self.is_initial_states)
      dist = torch.distributions.Categorical(probs=pi_hat)
      options = dist.sample()

      # storage next option data
      storage.add(prediction)
      storage.add(
          {'v': prediction['q_o'][self.worker_index, options].unsqueeze(-1)})
      storage.placeholder()

      self.compute_adv(storage)

    # training process
    self.learn(storage)
