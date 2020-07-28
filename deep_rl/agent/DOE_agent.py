#######################################################################
# Copyright (C) 2017 Shangtong Zhang(zhangshangtong.cpp@gmail.com)    #
# Permission given to modify the code as long as you keep this        #
# declaration at the top                                              #
#######################################################################

from ..network import *
from ..component import *
from .BaseAgent import *
from skimage import color


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

  def compute_pi_hat(self, prediction, prev_option, is_intial_states):
    # inter_pi/mask: [num_workers, num_options]
    inter_pi = prediction['inter_pi']
    mask = torch.zeros_like(inter_pi)
    # prev_option: [num_workers]
    mask[self.worker_index, prev_option] = 1
    # beta: [num_workers, num_options]
    beta = prediction['beta']
    # P(O_t|O_{t-1}, S_t) = \sum_{\beta} P(O_t, b_t| O_{t-1}, S_t)
    pi_hat = (1 - beta) * mask + beta * inter_pi
    # is_intial_states: [num_workers] ->
    # replicated to [num_workers, num_options]
    is_intial_states = is_intial_states.view(-1, 1).expand(-1, inter_pi.size(1))
    # pi_hat: [num_workers, num_options]; probability matrix for all options
    pi_hat = torch.where(is_intial_states, inter_pi, pi_hat)
    return pi_hat

  def compute_pi_bar(self, options, action, mean, std):
    # options: [num_workers, 1] -> [num_workers, 1, act_dim]
    options = options.unsqueeze(-1).expand(-1, -1, mean.size(-1))
    # mean: [num_workers, num_options, act_dim] -> [num_workers, act_dim]
    mean = mean.gather(1, options).squeeze(1)
    # mean: [num_workers, num_options, act_dim] -> [num_workers, act_dim]
    std = std.gather(1, options).squeeze(1)
    dist = torch.distributions.Normal(mean, std)
    pi_bar = dist.log_prob(action).sum(-1).exp().unsqueeze(-1)
    return pi_bar

  def compute_adv(self, storage):
    config = self.config

    v = storage.v
    adv = storage.adv
    all_ret = storage.ret

    with torch.no_grad():
      # ret: [num_workers, 1]
      ret = v[-1]
      # advantages: [num_workers, 1]
      advantages = tensor(np.zeros((config.num_workers, 1)))
      for i in reversed(range(config.rollout_length)):
        # m: [num_workers, 1]
        ret = storage.r[i] + config.discount * storage.m[i] * ret
        if not config.use_gae:
          advantages = ret - v[i]
        else:
          # td_error: [num_workers, 1]
          td_error = storage.r[i] +\
            config.discount * storage.m[i] * v[i+1] - v[i]
          advantages = advantages * config.gae_tau * config.discount *\
            storage.m[i] + td_error
        adv[i] = advantages
        all_ret[i] = ret

  def learn(self, storage):
    config = self.config

    states, actions, log_pi_bar_old, options, returns, advantages, inits, prev_options = storage.cat(
        ['s', 'a', 'log_pi_bar', 'o', 'ret', 'adv', 'init', 'prev_o'])
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
        sampled_states = states[batch_indices]
        sampled_actions = actions[batch_indices]
        sampled_options = options[batch_indices]
        sampled_log_pi_bar_old = log_pi_bar_old[batch_indices]
        sampled_returns = returns[batch_indices]
        sampled_advantages = advantages[batch_indices]
        sampled_inits = inits[batch_indices]
        sampled_prev_options = prev_options[batch_indices]

        prediction = self.network(sampled_states, sampled_prev_options)
        import ipdb; ipdb.set_trace(context=7)
        pi_bar = self.compute_pi_bar(sampled_options, sampled_actions,
                                     prediction['mean'], prediction['std'])
        log_pi_bar = pi_bar.add(1e-5).log()
        ratio = (log_pi_bar - sampled_log_pi_bar_old).exp()
        obj = ratio * sampled_advantages
        obj_clipped = ratio.clamp(
            1.0 - self.config.ppo_ratio_clip,
            1.0 + self.config.ppo_ratio_clip) * sampled_advantages
        policy_loss = -torch.min(obj, obj_clipped).mean()

        beta_adv = prediction['q_o'].gather(1, sampled_prev_options) - \
                   (prediction['q_o'] * prediction['inter_pi']).sum(-1).unsqueeze(-1)
        beta_adv = beta_adv.detach() + config.beta_reg
        beta_loss = prediction['beta'].gather(
            1, sampled_prev_options) * (1 - sampled_inits).float() * beta_adv
        beta_loss = beta_loss.mean()

        q_loss = (prediction['q_o'].gather(1, sampled_options) -
                  sampled_returns).pow(2).mul(0.5).mean()

        ent = -(prediction['log_inter_pi'] *
                prediction['inter_pi']).sum(-1).mean()
        inter_pi_loss = -(prediction['log_inter_pi'].gather(1, sampled_options) * sampled_advantages).mean()\
                        - config.entropy_weight * ent

        self.opt.zero_grad()
        (policy_loss + beta_loss + q_loss + inter_pi_loss).backward()
        nn.utils.clip_grad_norm_(self.network.parameters(),
                                 config.gradient_clip)
        self.opt.step()

  def rollout(self, storage, config, states):
    with torch.no_grad():
      for _ in range(config.rollout_length):
        prediction = self.network(states, self.prev_options.unsqueeze(1))
        ot = prediction['ot']

        self.logger.add_scalar('o_t', ot, log_level=5)
        self.logger.add_scalar('o_t_0', ot[0], log_level=5)
        self.logger.add_scalar(
            'pot_ent', prediction['pot_dist'].entropy(), log_level=5)
        self.logger.add_scalar(
            'pot_log_prob', prediction['pot_dist'].log_prob(ot), log_level=5)

        # mean/std: [num_workers, action_dim]
        mean = prediction['pat_mean']
        std = prediction['pat_std']
        dist = torch.distributions.Normal(mean, std)
        # actions: [num_workers, action_dim]
        actions = dist.sample()
        # pi_bar: [num_workers, 1]
        pi_bar = dist.log_prob(actions).sum(-1).exp().unsqueeze(-1)

        # next_states: tuple([state_dim] * num_workers)
        # terminals(bool)/rewards: [num_workers]
        # info: dict(['reward_run', 'reward_ctrl', 'episodic_return'] * 3)
        next_states, rewards, terminals, info = self.task.step(to_np(actions))
        self.record_online_return(info)
        rewards = config.reward_normalizer(rewards)
        # next_states: -> [num_workers, state_dim]
        next_states = config.state_normalizer(next_states)
        storage.add(prediction)
        '''
          r: [num_workers, 1]
          m: [num_workers, 1]
            0 for terminated states; 1 for continue
          a: [num_workers, act_dim]
          o: [num_workers, 1]
          prev_o: [num_workers, 1]
          s: [num_workers, state_dim]
          init: [num_workers, 1]
          v: [num_workers, 1]
          log_pi_bar: [num_workers, 1]
        '''
        storage.add({
            'r': tensor(rewards).unsqueeze(-1),
            'm': tensor(1 - terminals).unsqueeze(-1),
            'a': actions,
            'o': ot.unsqueeze(-1),
            'prev_o': self.prev_options.unsqueeze(-1),
            's': tensor(states),
            'init': self.is_initial_states.unsqueeze(-1),
            'v': prediction['q_ot_st'][self.worker_index, ot].unsqueeze(-1),
            'log_pi_bar': pi_bar.add(1e-5).log(),
        })

        # self.is_initial_states: [num_workers, 1]
        #    0 for continue; 1 for terminated
        self.is_initial_states = tensor(terminals).byte()
        self.prev_options = ot

        states = next_states
        self.total_steps += config.num_workers

      self.states = states
      prediction = self.network(states, self.prev_options.unsqueeze(-1))

      storage.add(prediction)
      # v: [num_workers, 1]
      storage.add({
          'v':
              prediction['q_ot_st']
              [self.worker_index, prediction['ot']].unsqueeze(-1)
      })
      storage.placeholder()

  def step(self):
    config = self.config
    storage = Storage(config.rollout_length)
    states = self.states
    self.rollout(storage, config, states)
    self.compute_adv(storage)
    self.learn(storage)
