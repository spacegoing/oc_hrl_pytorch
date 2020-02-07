#######################################################################
# Copyright (C) 2017 Shangtong Zhang(zhangshangtong.cpp@gmail.com)    #
# Permission given to modify the code as long as you keep this        #
# declaration at the top                                              #
#######################################################################

from .network_utils import *
from .network_bodies import *


class VanillaNet(BaseNet):

  def __init__(self, output_dim, body):
    super(VanillaNet, self).__init__()
    self.fc_head = layer_init(nn.Linear(body.feature_dim, output_dim))
    self.body = body
    self.to(Config.DEVICE)

  def forward(self, x):
    phi = self.body(tensor(x))
    y = self.fc_head(phi)
    return y


class DuelingNet(BaseNet):

  def __init__(self, action_dim, body):
    super(DuelingNet, self).__init__()
    self.fc_value = layer_init(nn.Linear(body.feature_dim, 1))
    self.fc_advantage = layer_init(nn.Linear(body.feature_dim, action_dim))
    self.body = body
    self.to(Config.DEVICE)

  def forward(self, x, to_numpy=False):
    phi = self.body(tensor(x))
    value = self.fc_value(phi)
    advantange = self.fc_advantage(phi)
    q = value.expand_as(advantange) + (
        advantange - advantange.mean(1, keepdim=True).expand_as(advantange))
    return q


class CategoricalNet(BaseNet):

  def __init__(self, action_dim, num_atoms, body):
    super(CategoricalNet, self).__init__()
    self.fc_categorical = layer_init(
        nn.Linear(body.feature_dim, action_dim * num_atoms))
    self.action_dim = action_dim
    self.num_atoms = num_atoms
    self.body = body
    self.to(Config.DEVICE)

  def forward(self, x):
    phi = self.body(tensor(x))
    pre_prob = self.fc_categorical(phi).view(
        (-1, self.action_dim, self.num_atoms))
    prob = F.softmax(pre_prob, dim=-1)
    log_prob = F.log_softmax(pre_prob, dim=-1)
    return prob, log_prob


class QuantileNet(BaseNet):

  def __init__(self, action_dim, num_quantiles, body):
    super(QuantileNet, self).__init__()
    self.fc_quantiles = layer_init(
        nn.Linear(body.feature_dim, action_dim * num_quantiles))
    self.action_dim = action_dim
    self.num_quantiles = num_quantiles
    self.body = body
    self.to(Config.DEVICE)

  def forward(self, x):
    phi = self.body(tensor(x))
    quantiles = self.fc_quantiles(phi)
    quantiles = quantiles.view((-1, self.action_dim, self.num_quantiles))
    return quantiles


class OptionCriticNet(BaseNet):

  def __init__(self, body, action_dim, num_options):
    super(OptionCriticNet, self).__init__()
    self.fc_q = layer_init(nn.Linear(body.feature_dim, num_options))
    self.fc_pi = layer_init(
        nn.Linear(body.feature_dim, num_options * action_dim))
    self.fc_beta = layer_init(nn.Linear(body.feature_dim, num_options))
    self.num_options = num_options
    self.action_dim = action_dim
    self.body = body
    self.to(Config.DEVICE)

  def forward(self, x):
    phi = self.body(tensor(x))
    q = self.fc_q(phi)
    beta = F.softmax(self.fc_beta(phi), dim=-1)
    pi = self.fc_pi(phi)
    pi = pi.view(-1, self.num_options, self.action_dim)
    log_pi = F.log_softmax(pi, dim=-1)
    pi = F.softmax(pi, dim=-1)
    return {'q': q, 'beta': beta, 'log_pi': log_pi, 'pi': pi}


class DeterministicActorCriticNet(BaseNet):

  def __init__(self,
               state_dim,
               action_dim,
               actor_opt_fn,
               critic_opt_fn,
               phi_body=None,
               actor_body=None,
               critic_body=None):
    super(DeterministicActorCriticNet, self).__init__()
    if phi_body is None:
      phi_body = DummyBody(state_dim)
    if actor_body is None:
      actor_body = DummyBody(phi_body.feature_dim)
    if critic_body is None:
      critic_body = DummyBody(phi_body.feature_dim)
    self.phi_body = phi_body
    self.actor_body = actor_body
    self.critic_body = critic_body
    self.fc_action = layer_init(
        nn.Linear(actor_body.feature_dim, action_dim), 1e-3)
    self.fc_critic = layer_init(nn.Linear(critic_body.feature_dim, 1), 1e-3)

    self.actor_params = list(self.actor_body.parameters()) + list(
        self.fc_action.parameters())
    self.critic_params = list(self.critic_body.parameters()) + list(
        self.fc_critic.parameters())
    self.phi_params = list(self.phi_body.parameters())

    self.actor_opt = actor_opt_fn(self.actor_params + self.phi_params)
    self.critic_opt = critic_opt_fn(self.critic_params + self.phi_params)
    self.to(Config.DEVICE)

  def forward(self, obs):
    phi = self.feature(obs)
    action = self.actor(phi)
    return action

  def feature(self, obs):
    obs = tensor(obs)
    return self.phi_body(obs)

  def actor(self, phi):
    return torch.tanh(self.fc_action(self.actor_body(phi)))

  def critic(self, phi, a):
    return self.fc_critic(self.critic_body(phi, a))


class GaussianActorCriticNet(BaseNet):

  def __init__(self,
               state_dim,
               action_dim,
               phi_body=None,
               actor_body=None,
               critic_body=None):
    super(GaussianActorCriticNet, self).__init__()
    if phi_body is None:
      phi_body = DummyBody(state_dim)
    if actor_body is None:
      actor_body = DummyBody(phi_body.feature_dim)
    if critic_body is None:
      critic_body = DummyBody(phi_body.feature_dim)
    self.phi_body = phi_body
    self.actor_body = actor_body
    self.critic_body = critic_body
    self.fc_action = layer_init(
        nn.Linear(actor_body.feature_dim, action_dim), 1e-3)
    self.fc_critic = layer_init(nn.Linear(critic_body.feature_dim, 1), 1e-3)

    self.actor_params = list(self.actor_body.parameters()) + list(
        self.fc_action.parameters())
    self.critic_params = list(self.critic_body.parameters()) + list(
        self.fc_critic.parameters())
    self.phi_params = list(self.phi_body.parameters())

    self.std = nn.Parameter(torch.zeros(action_dim))
    self.to(Config.DEVICE)

  def forward(self, obs, action=None):
    obs = tensor(obs)
    phi = self.phi_body(obs)
    phi_a = self.actor_body(phi)
    phi_v = self.critic_body(phi)
    mean = torch.tanh(self.fc_action(phi_a))
    v = self.fc_critic(phi_v)
    dist = torch.distributions.Normal(mean, F.softplus(self.std))
    if action is None:
      action = dist.sample()
    log_prob = dist.log_prob(action).sum(-1).unsqueeze(-1)
    entropy = dist.entropy().sum(-1).unsqueeze(-1)
    return {
        'a': action,
        'log_pi_a': log_prob,
        'ent': entropy,
        'mean': mean,
        'v': v
    }


class LstmActorCriticNet(BaseNet):

  def __init__(self,
               state_dim,
               action_dim,
               hid_dim,
               phi_body=None,
               actor_body=None,
               critic_body=None,
               config=None):
    super().__init__(config)
    self.is_recur = True
    self.action_dim = action_dim
    self.hid_dim = hid_dim
    if config.bi_direction:
      # h,c: (num_layers * num_directions, batch, hidden_size)
      self.hid_size = [config.num_lstm_layers * 2, None, self.hid_dim]
    else:
      self.hid_size = [config.num_lstm_layers, None, self.hid_dim]

    if phi_body is None:
      phi_body = DummyBody(state_dim)
    if critic_body is None:
      critic_body = DummyBody(config.lstm_to_fc_feat_dim)
    if actor_body is None:
      actor_body = DummyBody(config.lstm_to_fc_feat_dim)

    self.phi_body = phi_body
    self.actor_body = actor_body
    self.critic_body = critic_body

    self.lstm = lstm_init(
        nn.LSTM(
            phi_body.feature_dim,
            hid_dim,
            num_layers=config.num_lstm_layers,
            dropout=config.lstm_dropout,
            bidirectional=config.bi_direction), 1e-3)

    self.fc_action = layer_init(
        nn.Linear(self.actor_body.feature_dim, action_dim), 1e-3)
    self.fc_critic = layer_init(
        nn.Linear(self.critic_body.feature_dim, 1), 1e-3)

    # this is Config module, not self.config
    # device is selected select_device() in main
    self.std = nn.Parameter(torch.zeros(action_dim))
    self.to(Config.DEVICE)

  def forward(self, obs, input_lstm_states, masks, action_seq=None):
    '''
    obs: [timesteps, batch, feat_dim]
    input_lstm_states: (h, c) h/c: [num_layers * num_directions, batch, hidden_size]
    masks: [timesteps, batch]
    '''
    obs = tensor(obs)
    masks = tensor(masks)
    batch_size = masks.shape[1]
    # extends to [timesteps, batch, 1]
    # so it can be broadcast to [num_layers * num_directions, batch, hidden_size]
    # when multiplied with h and c
    masks = masks.unsqueeze(-1)
    phi = self.phi_body(obs)

    h_list = []
    h_input, c_input = input_lstm_states
    for p, m in zip(phi, masks):
      h_input = h_input * m
      c_input = c_input * m
      _, final_lstm_states = self.lstm(p.unsqueeze(0), (h_input, c_input))
      h_input, c_input = final_lstm_states
      h_list.append(h_input)
    # h,c: (num_layers * num_directions, batch, hidden_size)
    # (1,batch, hidden_size)

    a_list = []
    log_prob_list = []
    ent_list = []
    mean_list = []
    v_list = []
    for t, h in enumerate(h_list):
      # flat h into [batch, feat_dim] shape (ffn's input)
      # h: (num_layers * num_directions, batch, hidden_size)->
      #    (batch, hidden_size * num_layers * num_directions)
      h = h.permute([1, 0, 2]).reshape([batch_size, -1])

      phi_a = self.actor_body(h)
      phi_v = self.critic_body(h)
      mean = torch.tanh(self.fc_action(phi_a))
      v = self.fc_critic(phi_v)
      dist = torch.distributions.Normal(mean, F.softplus(self.std))
      # if action is not none, during PPO training stage
      # action [timesteps, action_dim]
      if action_seq is None:
        action = dist.sample()
      else:
        action = action_seq[t]
      log_prob = dist.log_prob(action).sum(-1).unsqueeze(-1)
      entropy = dist.entropy().sum(-1).unsqueeze(-1)

      a_list.append(action)
      log_prob_list.append(log_prob)
      ent_list.append(entropy)
      mean_list.append(mean)
      v_list.append(v)

    action, log_prob, h_final, entropy, mean, v = [
        torch.cat(i)
        for i in [a_list, log_prob_list, h_list, ent_list, mean_list, v_list]
    ]
    return {
        'a': action,
        'log_pi_a': log_prob,
        'input_lstm_states': input_lstm_states,
        'h_lstm_states': h_final,
        'final_lstm_states': final_lstm_states,
        'ent': entropy,
        'mean': mean,
        'v': v
    }

  def get_init_lstm_states(self, batchsize):
    # h,c: (num_layers * num_directions, batch, hidden_size)
    self.hid_size[1] = batchsize
    init_states = (torch.zeros(self.hid_size, device=Config.DEVICE),
                   torch.zeros(self.hid_size, device=Config.DEVICE))
    return init_states


class CategoricalActorCriticNet(BaseNet):

  def __init__(self,
               state_dim,
               action_dim,
               phi_body=None,
               actor_body=None,
               critic_body=None):
    super(CategoricalActorCriticNet, self).__init__()
    if phi_body is None:
      phi_body = DummyBody(state_dim)
    if actor_body is None:
      actor_body = DummyBody(phi_body.feature_dim)
    if critic_body is None:
      critic_body = DummyBody(phi_body.feature_dim)
    self.phi_body = phi_body
    self.actor_body = actor_body
    self.critic_body = critic_body
    self.fc_action = layer_init(
        nn.Linear(actor_body.feature_dim, action_dim), 1e-3)
    self.fc_critic = layer_init(nn.Linear(critic_body.feature_dim, 1), 1e-3)

    self.actor_params = list(self.actor_body.parameters()) + list(
        self.fc_action.parameters())
    self.critic_params = list(self.critic_body.parameters()) + list(
        self.fc_critic.parameters())
    self.phi_params = list(self.phi_body.parameters())

    self.to(Config.DEVICE)

  def forward(self, obs, action=None):
    obs = tensor(obs)
    phi = self.phi_body(obs)
    phi_a = self.actor_body(phi)
    phi_v = self.critic_body(phi)
    logits = self.fc_action(phi_a)
    v = self.fc_critic(phi_v)
    dist = torch.distributions.Categorical(logits=logits)
    if action is None:
      action = dist.sample()
    log_prob = dist.log_prob(action).unsqueeze(-1)
    entropy = dist.entropy().unsqueeze(-1)
    return {'a': action, 'log_pi_a': log_prob, 'ent': entropy, 'v': v}


class TD3Net(BaseNet):

  def __init__(
      self,
      action_dim,
      actor_body_fn,
      critic_body_fn,
      actor_opt_fn,
      critic_opt_fn,
  ):
    super(TD3Net, self).__init__()
    self.actor_body = actor_body_fn()
    self.critic_body_1 = critic_body_fn()
    self.critic_body_2 = critic_body_fn()

    self.fc_action = layer_init(
        nn.Linear(self.actor_body.feature_dim, action_dim), 1e-3)
    self.fc_critic_1 = layer_init(
        nn.Linear(self.critic_body_1.feature_dim, 1), 1e-3)
    self.fc_critic_2 = layer_init(
        nn.Linear(self.critic_body_2.feature_dim, 1), 1e-3)

    self.actor_params = list(self.actor_body.parameters()) + list(
        self.fc_action.parameters())
    self.critic_params = list(self.critic_body_1.parameters()) + list(self.fc_critic_1.parameters()) +\
                         list(self.critic_body_2.parameters()) + list(self.fc_critic_2.parameters())

    self.actor_opt = actor_opt_fn(self.actor_params)
    self.critic_opt = critic_opt_fn(self.critic_params)
    self.to(Config.DEVICE)

  def forward(self, obs):
    obs = tensor(obs)
    return torch.tanh(self.fc_action(self.actor_body(obs)))

  def q(self, obs, a):
    obs = tensor(obs)
    a = tensor(a)
    x = torch.cat([obs, a], dim=1)
    q_1 = self.fc_critic_1(self.critic_body_1(x))
    q_2 = self.fc_critic_2(self.critic_body_2(x))
    return q_1, q_2


class OptionGaussianActorCriticNet(BaseNet):

  def __init__(self,
               state_dim,
               action_dim,
               num_options,
               phi_body=None,
               actor_body=None,
               critic_body=None,
               option_body_fn=None):
    super().__init__()
    if phi_body is None:
      phi_body = DummyBody(state_dim)
    if critic_body is None:
      critic_body = DummyBody(phi_body.feature_dim)
    if actor_body is None:
      actor_body = DummyBody(phi_body.feature_dim)

    self.phi_body = phi_body
    self.actor_body = actor_body
    self.critic_body = critic_body

    # build option network
    self.options = nn.ModuleList([
        SingleOptionNet(action_dim, option_body_fn) for _ in range(num_options)
    ])

    # linear output
    self.fc_pi_o = layer_init(
        nn.Linear(actor_body.feature_dim, num_options), 1e-3)
    self.fc_q_o = layer_init(
        nn.Linear(critic_body.feature_dim, num_options), 1e-3)

    self.num_options = num_options
    self.action_dim = action_dim
    self.to(Config.DEVICE)

  def forward(self, obs):

    # state feature
    obs = tensor(obs)
    phi = self.phi_body(obs)

    # option
    mean = []
    std = []
    beta = []
    for option in self.options:
      prediction = option(phi)
      mean.append(prediction['mean'].unsqueeze(1))
      std.append(prediction['std'].unsqueeze(1))
      beta.append(prediction['beta'])
    mean = torch.cat(mean, dim=1)
    std = torch.cat(std, dim=1)
    beta = torch.cat(beta, dim=1)

    # policy over option with soft-max
    phi_a = self.actor_body(phi)
    phi_a = self.fc_pi_o(phi_a)
    pi_o = F.softmax(phi_a, dim=-1)
    log_pi_o = F.log_softmax(phi_a, dim=-1)

    # critic network
    phi_c = self.critic_body(phi)
    q_o = self.fc_q_o(phi_c)

    return {
        'mean': mean,
        'std': std,
        'q_o': q_o,
        'pi_o': pi_o,
        'log_pi_o': log_pi_o,
        'beta': beta
    }


class LstmOptionGaussianActorCriticNet(BaseNet):

  def __init__(self,
               state_dim,
               action_dim,
               num_options,
               hid_dim,
               phi_body=None,
               actor_body=None,
               critic_body=None,
               option_body_fn=None,
               config=None):
    super().__init__(config)
    self.is_recur = True

    if phi_body is None:
      phi_body = DummyBody(state_dim)
    if critic_body is None:
      critic_body = DummyBody(phi_body.feature_dim)
    if actor_body is None:
      actor_body = DummyBody(phi_body.feature_dim)

    self.phi_body = phi_body
    self.actor_body = actor_body
    self.critic_body = critic_body

    # build option network
    self.options = nn.ModuleList([
        SingleLstmOptionNet(action_dim, hid_dim, option_body_fn, config)
        for _ in range(num_options)
    ])

    # linear output
    self.lstm_pi_o = lstm_init(
        nn.LSTM(actor_body.feature_dim, hid_dim, batch_first=True), 1e-3)
    self.fc_pi_o = layer_init(nn.Linear(hid_dim, num_options), 1e-3)

    self.lstm_q_o = lstm_init(
        nn.LSTM(critic_body.feature_dim, hid_dim, batch_first=True), 1e-3)
    self.fc_q_o = layer_init(nn.Linear(hid_dim, num_options), 1e-3)

    self.num_options = num_options
    self.action_dim = action_dim
    self.hid_dim = hid_dim
    # this is Config module, not self.config
    # device is selected select_device() in main
    self.to(Config.DEVICE)

  def forward(self, obs, pi_o_final_states, q_o_final_states,
              all_option_final_states, prev_options, masks):
    '''
    pi_o_states: [(seq_len, batchsize, hid_dim), (seq_len, batchsize, hid_dim)]
    q_o_states: [(seq_len, batchsize, hid_dim), (seq_len, batchsize, hid_dim)]
    option_states: [pi_a_states, beta_states]
    prev_option: int. prev selected option

    other than prev_option using option_states, other options using zero states

    returns:
    all_option_final_states: list:[num_options] each entry: [pi_a_final_state, beta_final_state]
    '''

    obs = tensor(obs)

    def _one_step_forward(obs, pi_o_final_states, q_o_final_states,
                          all_option_final_states, prev_options, batchsize,
                          masks):
      all_option_final_states = self.assign_option_init_states(
          prev_options, all_option_final_states, batchsize, masks)
      pi_o_final_states = self.get_masked_manager_lstm_states(
          pi_o_final_states, masks)
      q_o_final_states = self.get_masked_manager_lstm_states(
          q_o_final_states, masks)
      phi = self.phi_body(obs)

      # option
      mean = []
      std = []
      beta = []
      new_option_final_states = []

      for option_idx, [pi_a_final_states,
                       beta_final_states] in enumerate(all_option_final_states):
        prediction = self.options[option_idx](phi, pi_a_final_states,
                                              beta_final_states)
        mean.append(prediction['mean'].unsqueeze(1))
        std.append(prediction['std'].unsqueeze(1))
        beta.append(prediction['beta'])
        new_option_final_states.append(
            (prediction['pi_a_final_state'], prediction['beta_final_state']))
      mean = torch.cat(mean, dim=1)
      std = torch.cat(std, dim=1)
      beta = torch.cat(beta, dim=1)

      # policy over option with soft-max
      phi_a = self.actor_body(phi)
      # final_states = (hid_state, cell_state)
      _, pi_o_final_states = self.lstm_pi_o(
          phi_a.unsqueeze(1), pi_o_final_states)
      pi_o_hid_state = pi_o_final_states[0]
      # pi_o_hid_state [seq_len, batchsize, hid_dim]
      # here only have 1 timestep, use [0] to squeeze
      phi_a = self.fc_pi_o(pi_o_hid_state[0])
      pi_o = F.softmax(phi_a, dim=-1)
      log_pi_o = F.log_softmax(phi_a, dim=-1)

      # critic network
      phi_c = self.critic_body(phi)
      _, q_o_final_states = self.lstm_q_o(phi_c.unsqueeze(1), q_o_final_states)
      q_o_hid_state = q_o_final_states[0]
      q_o = self.fc_q_o(q_o_hid_state[0])

      return {
          'mean': mean,
          'std': std,
          'beta': beta,
          'pi_o': pi_o,
          'log_pi_o': log_pi_o,
          'pi_o_final_states': pi_o_final_states,
          'q_o': q_o,
          'q_o_final_states': q_o_final_states,
          'all_option_final_states': new_option_final_states
      }

    if len(obs.shape) == 2:
      return _one_step_forward(obs, pi_o_final_states, q_o_final_states,
                               all_option_final_states, prev_options,
                               self.config.num_workers, masks)
    else:
      seq_len = obs.shape[1]
      result_dict = {
          'mean': [],
          'std': [],
          'beta': [],
          'pi_o': [],
          'log_pi_o': [],
          'q_o': []
      }
      for i in range(seq_len):
        prediction = _one_step_forward(obs[:, i, :], pi_o_final_states,
                                       q_o_final_states,
                                       all_option_final_states,
                                       prev_options[:, i, :], obs.shape[0],
                                       masks[:, i, :])
        for k in result_dict:
          result_dict[k].append(prediction[k])
        pi_o_final_states, q_o_final_states,\
        all_option_final_states = prediction['pi_o_final_states'],\
                                  prediction['q_o_final_states'],\
                                  prediction['all_option_final_states']
      for k in result_dict:
        result_dict[k] = torch.stack(
            result_dict[k], dim=1).view(prediction[k].shape[0] * seq_len,
                                        *prediction[k].shape[1:])
      result_dict.update({
          'pi_o_final_states': pi_o_final_states,
          'q_o_final_states': q_o_final_states,
          'all_option_final_states': all_option_final_states
      })
      return result_dict

  def get_single_option_init_states(self, batchsize):
    pi_a_states = (torch.zeros([1, batchsize, self.hid_dim],
                               device=Config.DEVICE),
                   torch.zeros([1, batchsize, self.hid_dim],
                               device=Config.DEVICE))
    beta_states = (torch.zeros([1, batchsize, self.hid_dim],
                               device=Config.DEVICE),
                   torch.zeros([1, batchsize, self.hid_dim],
                               device=Config.DEVICE))
    option_states = (pi_a_states, beta_states)
    return option_states

  def get_manager_option_q_init_states(self, batchsize):
    pi_o_states = (torch.zeros([1, batchsize, self.hid_dim],
                               device=Config.DEVICE),
                   torch.zeros([1, batchsize, self.hid_dim],
                               device=Config.DEVICE))
    q_o_states = (torch.zeros([1, batchsize, self.hid_dim],
                              device=Config.DEVICE),
                  torch.zeros([1, batchsize, self.hid_dim],
                              device=Config.DEVICE))
    return pi_o_states, q_o_states

  def get_all_init_states(self, batchsize):
    pi_o_hid_states, q_o_hid_states = self.get_manager_option_q_init_states(
        batchsize)
    all_option_states = [
        self.get_single_option_init_states(batchsize)
        for i in range(self.num_options)
    ]
    return pi_o_hid_states, q_o_hid_states, all_option_states

  def assign_option_init_states(self, prev_options, prev_option_final_states,
                                batchsize, masks):
    all_option_final_states = [
        self.get_single_option_init_states(batchsize)
        for i in range(self.num_options)
    ]
    for worker_idx, option_idx in enumerate(prev_options):
      pi_a_final_states, beta_final_states = all_option_final_states[option_idx]
      prev_pi_a_final_states, prev_beta_final_states = prev_option_final_states[
          option_idx]
      # assign pi_a hid, cell states
      pi_a_hid_state, pi_a_cell_state = pi_a_final_states
      prev_pi_a_hid_state, prev_pi_a_cell_state = prev_pi_a_final_states
      pi_a_hid_state[:,
                     worker_idx, :] = prev_pi_a_hid_state[:,
                                                          worker_idx, :] * masks[
                                                              worker_idx]
      pi_a_cell_state[:,
                      worker_idx, :] = prev_pi_a_cell_state[:,
                                                            worker_idx, :] * masks[
                                                                worker_idx]
      # assign beta hid, cell states
      beta_hid_state, beta_cell_state = beta_final_states
      prev_beta_hid_state, prev_beta_cell_state = prev_beta_final_states
      beta_hid_state[:,
                     worker_idx, :] = prev_beta_hid_state[:,
                                                          worker_idx, :] * masks[
                                                              worker_idx]
      beta_cell_state[:,
                      worker_idx, :] = prev_beta_cell_state[:,
                                                            worker_idx, :] * masks[
                                                                worker_idx]
    return all_option_final_states

  def get_masked_manager_lstm_states(self, hid_cell_states, masks):
    masks = masks.view(1, masks.shape[0], 1)
    return hid_cell_states[0] * masks, hid_cell_states[1] * masks


class SoftOptionGaussianActorCriticNet(BaseNet):

  def __init__(self,
               state_dim,
               action_dim,
               num_options,
               phi_body=None,
               actor_body=None,
               action_critic_body_1=None,
               action_critic_body_2=None,
               critic_body=None,
               option_body_fn=None):
    super(SoftOptionGaussianActorCriticNet, self).__init__()

    if phi_body is None:
      phi_body = DummyBody(state_dim)
    if actor_body is None:
      actor_body = DummyBody(phi_body.feature_dim)

    # calculate one state-option function
    if critic_body is None:
      critic_body = DummyBody(phi_body.feature_dim)

    # calculate two state-option-action function
    if action_critic_body_1 is None:
      action_critic_body_1 = DummyBody(phi_body.feature_dim)
    if action_critic_body_2 is None:
      action_critic_body_2 = DummyBody(phi_body.feature_dim)

    self.phi_body = phi_body
    self.actor_body = actor_body
    self.critic_body = critic_body
    self.action_critic_body_1 = action_critic_body_1
    self.action_critic_body_2 = action_critic_body_2

    # build option network
    self.options = nn.ModuleList([
        SingleOptionNet(action_dim, option_body_fn) for _ in range(num_options)
    ])

    # critic network :: output layer
    self.fc_pi_o = layer_init(
        nn.Linear(self.actor_body.feature_dim, num_options), 1e-3)
    self.fc_q_o_1 = layer_init(
        nn.Linear(self.action_critic_body_1.feature_dim, num_options), 1e-3)
    self.fc_q_o_2 = layer_init(
        nn.Linear(self.action_critic_body_2.feature_dim, num_options), 1e-3)
    self.fc_v_o = layer_init(
        nn.Linear(self.critic_body.feature_dim, num_options), 1e-3)

    self.fc_u_o = layer_init(
        nn.Linear(critic_body.feature_dim, num_options + 1), 1e-3)

    self.num_options = num_options
    self.action_dim = action_dim
    self.to(Config.DEVICE)

  def forward(self, obs):

    # state feature
    obs = tensor(obs)
    phi = self.phi_body(obs)

    # option
    mean = []
    std = []
    beta = []
    for option in self.options:
      prediction = option(phi)
      mean.append(prediction['mean'].unsqueeze(1))
      std.append(prediction['std'].unsqueeze(1))
      beta.append(prediction['beta'])
    mean = torch.cat(mean, dim=1)
    std = torch.cat(std, dim=1)
    beta = torch.cat(beta, dim=1)

    # policy over option
    phi_a = self.actor_body(phi)
    phi_a = self.fc_pi_o(phi_a)
    pi_o = F.softmax(phi_a, dim=-1)
    log_pi_o = F.log_softmax(phi_a, dim=-1)

    # critic
    phi_c = self.critic_body(phi)
    q_o_1 = self.fc_q_o_1(phi_c)
    q_o_2 = self.fc_q_o_2(phi_c)
    v_o = self.fc_v_o(phi_c)
    u_o = self.fc_u_o(phi_c)

    return {
        'mean': mean,
        'std': std,
        'q_o_1': q_o_1,
        'q_o_2': q_o_2,
        'v_o': v_o,
        'u_o': u_o,
        'pi_o': pi_o,
        'log_pi_o': log_pi_o,
        'beta': beta
    }


class InterOptionPGNet(BaseNet):

  def __init__(self, body, action_dim, num_options):
    super(InterOptionPGNet, self).__init__()
    self.fc_q = layer_init(nn.Linear(body.feature_dim, num_options))
    self.fc_pi_o = layer_init(nn.Linear(body.feature_dim, num_options))
    self.fc_pi = layer_init(
        nn.Linear(body.feature_dim, num_options * action_dim))
    self.fc_beta = layer_init(nn.Linear(body.feature_dim, num_options))
    self.num_options = num_options
    self.action_dim = action_dim
    self.body = body
    self.to(Config.DEVICE)

  def forward(self, x, phi=None):
    if phi is None:
      phi = self.body(tensor(x))
    q = self.fc_q(phi)
    beta = torch.sigmoid(self.fc_beta(phi))
    pi = self.fc_pi(phi)
    pi = pi.view(-1, self.num_options, self.action_dim)
    log_pi = F.log_softmax(pi, dim=-1)
    pi = F.softmax(pi, dim=-1)

    pi_o = self.fc_pi_o(phi)
    log_pi_o = F.log_softmax(pi_o, dim=-1)
    pi_o = F.softmax(pi_o, dim=-1)

    return {
        'q': q,
        'beta': beta,
        'log_pi': log_pi,
        'pi': pi,
        'log_pi_o': log_pi_o,
        'pi_o': pi_o,
        'phi': phi
    }


class SingleOptionNet(nn.Module):

  def __init__(self, action_dim, body_fn):
    super(SingleOptionNet, self).__init__()
    self.pi_body = body_fn()
    self.beta_body = body_fn()
    self.fc_pi = layer_init(
        nn.Linear(self.pi_body.feature_dim, action_dim), 1e-3)
    self.fc_beta = layer_init(nn.Linear(self.beta_body.feature_dim, 1), 1e-3)
    self.std = nn.Parameter(torch.zeros((1, action_dim)))

  def forward(self, phi):
    phi_pi = self.pi_body(phi)
    mean = F.tanh(self.fc_pi(phi_pi))
    std = F.softplus(self.std).expand(mean.size(0), -1)

    phi_beta = self.beta_body(phi)
    beta = torch.sigmoid(self.fc_beta(phi_beta))

    return {
        'mean': mean,
        'std': std,
        'beta': beta,
    }


class SingleLstmOptionNet(BaseNet):

  def __init__(self, action_dim, hid_dim, body_fn, config):
    super().__init__(config)
    self.pi_body = body_fn()
    self.beta_body = body_fn()
    self.lstm_pi = lstm_init(
        nn.LSTM(self.pi_body.feature_dim, hid_dim, batch_first=True), 1e-3)
    self.lstm_beta = lstm_init(
        nn.LSTM(self.beta_body.feature_dim, hid_dim, batch_first=True), 1e-3)
    self.fc_pi = layer_init(nn.Linear(hid_dim, action_dim), 1e-3)
    self.fc_beta = layer_init(nn.Linear(hid_dim, 1), 1e-3)
    self.std = nn.Parameter(torch.zeros((1, action_dim)))
    self.is_recur = True

  def forward(self, phi, pi_a_state, beta_state):
    '''
    pi_a_state: [(seq_len, batchsize, hid_dim), (seq_len, batchsize, hid_dim)]
    beta_state: [(seq_len, batchsize, hid_dim), (seq_len, batchsize, hid_dim)]

    for ppo, seq_len = 1
    '''
    phi_pi = self.pi_body(phi)
    _, pi_a_final_state = self.lstm_pi(phi_pi.unsqueeze(1), pi_a_state)
    pi_a_hid_state = pi_a_final_state[0]
    # pi_a_hid_state [seq_len, batchsize, hid_dim]
    # here only have 1 timestep, use [0] to squeeze
    mean = torch.tanh(self.fc_pi(pi_a_hid_state[0]))
    std = F.softplus(self.std).expand(mean.size(0), -1)

    phi_beta = self.beta_body(phi)
    _, beta_final_state = self.lstm_beta(phi_beta.unsqueeze(1), beta_state)
    beta_hid_state = beta_final_state[0]
    beta = F.softmax(self.fc_beta(beta_hid_state[0]), dim=-1)

    return {
        'mean': mean,
        'pi_a_final_state': pi_a_final_state,
        'std': std,
        'beta': beta,
        'beta_final_state': beta_final_state
    }


class DeterministicOptionCriticNet(BaseNet):

  def __init__(self,
               action_dim,
               phi_body,
               actor_body,
               critic_body,
               beta_body,
               num_options,
               actor_opt_fn,
               critic_opt_fn,
               gpu=-1):

    super(DeterministicOptionCriticNet, self).__init__()
    self.phi_body = phi_body
    self.actor_body = actor_body
    self.critic_body = critic_body
    self.beta_body = beta_body
    self.action_dim = action_dim
    self.num_options = num_options

    self.fc_actors = nn.ModuleList([
        layer_init(nn.Linear(actor_body.feature_dim, action_dim))
        for _ in range(num_options)
    ])
    self.fc_beta = layer_init(nn.Linear(beta_body.feature_dim, num_options))
    self.fc_critics = nn.ModuleList([
        layer_init(nn.Linear(critic_body.feature_dim, 1))
        for _ in range(num_options)
    ])

    self.actor_params = list(self.actor_body.parameters()) + list(
        self.fc_actors.parameters())
    self.critic_params = list(self.critic_body.parameters()) + list(
        self.fc_critics.parameters())
    self.beta_params = list(self.beta_body.parameters()) + list(
        self.fc_beta.parameters())
    self.phi_params = list(self.phi_body.parameters())

    self.actor_opt = actor_opt_fn(self.actor_params + self.phi_params)
    self.critic_opt = critic_opt_fn(self.critic_params + self.phi_params +
                                    self.beta_params)

    # self.set_gpu(gpu)

  def feature(self, obs):
    obs = self.tensor(obs)
    phi = self.phi_body(obs)
    return phi

  def predict(self, obs, to_numpy=False):
    phi = self.feature(obs)
    actions = self.actor(phi)
    betas = self.termination(phi)
    q_values = self.critic(phi, actions)
    actions = torch.stack(actions).transpose(0, 1)
    return q_values, betas, actions

  def termination(self, phi):
    phi_beta = self.beta_body(phi)
    beta = torch.sigmoid(self.fc_beta(phi_beta))
    return beta

  def actor(self, phi):
    phi_actor = self.actor_body(phi)
    actions = [F.tanh(fc_actor(phi_actor)) for fc_actor in self.fc_actors]
    return actions

  def critic(self, phi, actions):
    if isinstance(actions, torch.Tensor):
      phi = self.critic_body(phi, actions)
      q = [fc_critic(phi) for fc_critic in self.fc_critics]
    elif isinstance(actions, list):
      q = [
          fc_critic(self.critic_body(phi, action))
          for fc_critic, action in zip(self.fc_critics, actions)
      ]
    q = torch.cat(q, dim=1)
    return q


class GammaDeterministicOptionCriticNet(BaseNet):

  def __init__(self,
               action_dim,
               phi_body,
               actor_body,
               critic_body,
               num_options,
               gpu=-1):
    super(GammaDeterministicOptionCriticNet, self).__init__()
    self.phi_body = phi_body
    self.actor_body = actor_body
    self.critic_body = critic_body
    self.action_dim = action_dim
    self.num_options = num_options

    self.fc_q_options = layer_init(
        nn.Linear(actor_body.feature_dim, num_options))
    self.fc_actors = nn.ModuleList([
        layer_init(nn.Linear(actor_body.feature_dim, action_dim))
        for _ in range(num_options)
    ])
    self.fc_critics = nn.ModuleList([
        layer_init(nn.Linear(critic_body.feature_dim, 1))
        for _ in range(num_options)
    ])

    self.set_gpu(gpu)

  def feature(self, obs):
    obs = self.tensor(obs)
    phi = self.phi_body(obs)
    return phi

  def predict(self, obs, to_numpy=False):
    phi = self.feature(obs)
    actions, q_options = self.actor(phi)
    # q_values = self.critic(phi, actions)
    # actions = torch.stack(actions).transpose(0, 1)
    # best = q_values.max(1)[1]
    # if to_numpy:
    #     actions = actions[self.tensor(np.arange(actions.size(0))).long(), best, :]
    #     return actions.detach().cpu().numpy(), best.detach().cpu().numpy()
    # return actions, q_values, best
    return actions, q_options

  def actor(self, phi):
    phi_actor = self.actor_body(phi)
    actions = [F.tanh(fc_actor(phi_actor)) for fc_actor in self.fc_actors]
    q_options = self.fc_q_options(phi_actor)
    return actions, q_options

  def critic(self, phi, actions):
    if isinstance(actions, torch.Tensor):
      phi = self.critic_body(phi, actions)
      q = [fc_critic(phi) for fc_critic in self.fc_critics]
    elif isinstance(actions, list):
      q = [
          fc_critic(self.critic_body(phi, action))
          for fc_critic, action in zip(self.fc_critics, actions)
      ]
    q = torch.cat(q, dim=-1)
    return q

  def zero_non_actor_grad(self):
    self.fc_q_options.zero_grad()
    self.fc_critics.zero_grad()
    self.critic_body.zero_grad()


class EnvModel(nn.Module):

  def __init__(self, phi_dim, action_dim):
    super(EnvModel, self).__init__()
    self.hidden_dim = 300
    self.fc_r1 = layer_init(nn.Linear(phi_dim + action_dim, self.hidden_dim))
    self.fc_r2 = layer_init(nn.Linear(self.hidden_dim, 1))

    self.fc_t1 = layer_init(nn.Linear(phi_dim, phi_dim))
    self.fc_t2 = layer_init(nn.Linear(phi_dim + action_dim, phi_dim))

  def forward(self, phi_s, action):
    phi = torch.cat([phi_s, action], dim=-1)
    r = self.fc_r2(F.tanh(self.fc_r1(phi)))

    phi_s_prime = phi_s + F.tanh(self.fc_t1(phi_s))
    phi_sa_prime = torch.cat([phi_s_prime, action], dim=-1)
    phi_s_prime = phi_s_prime + F.tanh(self.fc_t2(phi_sa_prime))

    return phi_s_prime, r


class ActorModel(nn.Module):

  def __init__(self, phi_dim, action_dim):
    super(ActorModel, self).__init__()
    self.hidden_dim = 300
    self.layers = nn.Sequential(
        layer_init(nn.Linear(phi_dim, self.hidden_dim)), nn.Tanh(),
        layer_init(nn.Linear(self.hidden_dim, action_dim), 3e-3), nn.Tanh())

  def forward(self, phi_s):
    return self.layers(phi_s)


class CriticModel(nn.Module):

  def __init__(self, phi_dim, action_dim):
    super(CriticModel, self).__init__()
    self.hidden_dim = 300
    self.layers = nn.Sequential(
        layer_init(nn.Linear(phi_dim + action_dim, self.hidden_dim)), nn.Tanh(),
        layer_init(nn.Linear(self.hidden_dim, 1), 3e-3))

  def forward(self, phi_s, action):
    phi = torch.cat([phi_s, action], dim=-1)
    return self.layers(phi)


class PlanEnsembleDeterministicNet(BaseNet):

  def __init__(self,
               state_dim,
               action_dim,
               phi_body,
               num_actors,
               discount,
               detach_action,
               gpu=-1):
    super(PlanEnsembleDeterministicNet, self).__init__()
    self.phi_body = phi_body
    phi_dim = phi_body.feature_dim
    self.critic_model = CriticModel(phi_dim, action_dim)
    self.actor_models = nn.ModuleList(
        [ActorModel(phi_dim, action_dim) for _ in range(num_actors)])
    self.env_model = EnvModel(phi_dim, action_dim)

    self.discount = discount
    self.detach_action = detach_action
    self.num_actors = num_actors
    self.set_gpu(gpu)

  def predict(self, obs, depth, to_numpy=False):
    phi = self.feature(obs)
    actions = self.compute_a(phi)
    q_values = [self.compute_q(phi, action, depth) for action in actions]
    q_values = torch.stack(q_values).squeeze(-1).t()
    actions = torch.stack(actions).transpose(0, 1)
    if to_numpy:
      best = q_values.max(1)[1]
      actions = actions[self.range(actions.size(0)), best, :]
      return actions.detach().cpu().numpy(), best.detach().cpu().numpy()
    return q_values.max(1)[0].unsqueeze(-1)

  def feature(self, obs):
    obs = self.tensor(obs)
    return self.phi_body(obs)

  def compute_a(self, phi, detach=True):
    actions = [actor_model(phi) for actor_model in self.actor_models]
    if detach:
      for action in actions:
        action.detach_()
    return actions

  def compute_q(self, phi, action, depth=1, immediate_reward=False):
    if depth == 1:
      q = self.critic_model(phi, action)
      if immediate_reward:
        return q, 0
      return q
    else:
      phi_prime, r = self.env_model(phi, action)
      a_prime = self.compute_a(phi_prime)
      a_prime = torch.stack(a_prime)
      phi_prime = phi_prime.unsqueeze(0).expand((self.num_actors,) +
                                                (-1,) * len(phi_prime.size()))
      q_prime = self.compute_q(phi_prime, a_prime, depth - 1)
      q_prime = q_prime.max(0)[0]
      q = r + self.discount * q_prime
      if immediate_reward:
        return q, r
      return q

  def actor(self, obs):
    phi = self.compute_phi(obs)
    actions = self.compute_a(phi, detach=False)
    return actions


class NaiveModelDDPGNet(BaseNet):

  def __init__(self,
               state_dim,
               action_dim,
               phi_body,
               num_actors,
               discount,
               detach_action,
               gpu=-1):
    super(NaiveModelDDPGNet, self).__init__()
    self.phi_body = phi_body
    phi_dim = phi_body.feature_dim
    self.critic_model = CriticModel(phi_dim, action_dim)
    self.actor_models = nn.ModuleList(
        [ActorModel(phi_dim, action_dim) for _ in range(num_actors)])
    self.env_model = EnvModel(phi_dim, action_dim)

    self.discount = discount
    self.detach_action = detach_action
    self.num_actors = num_actors
    self.set_gpu(gpu)

  def predict(self, obs, depth, to_numpy=False):
    phi = self.feature(obs)
    actions = self.compute_a(phi)
    q_values = [self.compute_q(phi, action, depth) for action in actions]
    q_values = torch.stack(q_values).squeeze(-1).t()
    actions = torch.stack(actions).t()
    if to_numpy:
      best = q_values.max(1)[1]
      actions = actions[self.range(actions.size(0)), best, :]
      return actions.detach().cpu().numpy()
    return q_values.max(1)[0].unsqueeze(-1)

  def feature(self, obs):
    obs = self.tensor(obs)
    return self.phi_body(obs)

  def compute_a(self, phi, detach=True):
    actions = [actor_model(phi) for actor_model in self.actor_models]
    if detach:
      for action in actions:
        action.detach_()
    return actions

  def compute_q(self, phi, action, depth=1, immediate_reward=False):
    if depth == 1:
      q = self.critic_model(phi, action)
      if immediate_reward:
        return q, 0
      return q
    else:
      phi_prime, r = self.env_model(phi, action)
      a_prime = self.compute_a(phi_prime)
      a_prime = torch.stack(a_prime)
      phi_prime = phi_prime.unsqueeze(0).expand((self.num_actors,) +
                                                (-1,) * len(phi_prime.size()))
      q_prime = self.compute_q(phi_prime, a_prime, depth - 1)
      q_prime = q_prime.max(0)[0]
      q = r + self.discount * q_prime
      if immediate_reward:
        return q, r
      return q

  def actor(self, obs):
    phi = self.compute_phi(obs)
    actions = self.compute_a(phi, detach=False)
    return actions
