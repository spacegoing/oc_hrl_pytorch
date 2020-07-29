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
    beta = F.sigmoid(self.fc_beta(phi))
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
    return F.tanh(self.fc_action(self.actor_body(phi)))

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
    mean = F.tanh(self.fc_action(phi_a))
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


class InterOptionPGNet(BaseNet):

  def __init__(self, body, action_dim, num_options):
    super(InterOptionPGNet, self).__init__()
    self.fc_q = layer_init(nn.Linear(body.feature_dim, num_options))
    self.fc_inter_pi = layer_init(nn.Linear(body.feature_dim, num_options))
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
    beta = F.sigmoid(self.fc_beta(phi))
    pi = self.fc_pi(phi)
    pi = pi.view(-1, self.num_options, self.action_dim)
    log_pi = F.log_softmax(pi, dim=-1)
    pi = F.softmax(pi, dim=-1)

    inter_pi = self.fc_inter_pi(phi)
    log_inter_pi = F.log_softmax(inter_pi, dim=-1)
    inter_pi = F.softmax(inter_pi, dim=-1)

    return {
        'q': q,
        'beta': beta,
        'log_pi': log_pi,
        'pi': pi,
        'log_inter_pi': log_inter_pi,
        'inter_pi': inter_pi,
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
    beta = F.sigmoid(self.fc_beta(phi_beta))

    return {
        'mean': mean,
        'std': std,
        'beta': beta,
    }


class OptionGaussianActorCriticNet(BaseNet):

  def __init__(self,
               state_dim,
               action_dim,
               num_options,
               phi_body=None,
               actor_body=None,
               critic_body=None,
               option_body_fn=None):
    super(OptionGaussianActorCriticNet, self).__init__()
    if phi_body is None:
      phi_body = DummyBody(state_dim)
    if critic_body is None:
      critic_body = DummyBody(phi_body.feature_dim)
    if actor_body is None:
      actor_body = DummyBody(phi_body.feature_dim)

    self.phi_body = phi_body
    self.actor_body = actor_body
    self.critic_body = critic_body

    self.options = nn.ModuleList([
        SingleOptionNet(action_dim, option_body_fn) for _ in range(num_options)
    ])

    self.fc_pi_o = layer_init(
        nn.Linear(actor_body.feature_dim, num_options), 1e-3)
    self.fc_q_o = layer_init(
        nn.Linear(critic_body.feature_dim, num_options), 1e-3)
    self.fc_u_o = layer_init(
        nn.Linear(critic_body.feature_dim, num_options + 1), 1e-3)

    self.num_options = num_options
    self.action_dim = action_dim
    self.to(Config.DEVICE)

  def forward(self, obs):
    '''
    Params:
        obs: [num_workers, state_dim]

    Returns:
        inter_pi: [num_workers, num_options]
        log_inter_pi: [num_workers, num_options]
        beta: [num_workers, num_options]
        mean: [num_workers, num_options, action_dim]
        std: [num_workers, num_options, action_dim]
        q_o: [num_workers, num_options]
        u_o: [num_workers, num_options+1]
    '''
    obs = tensor(obs)
    phi = self.phi_body(obs)

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

    phi_a = self.actor_body(phi)
    phi_a = self.fc_pi_o(phi_a)
    pi_o = F.softmax(phi_a, dim=-1)
    log_pi_o = F.log_softmax(phi_a, dim=-1)

    phi_c = self.critic_body(phi)
    q_o = self.fc_q_o(phi_c)
    u_o = self.fc_u_o(phi_c)

    return {
        'mean': mean,
        'std': std,
        'q_o': q_o,
        'u_o': u_o,
        'inter_pi': pi_o,
        'log_inter_pi': log_pi_o,
        'beta': beta
    }


class DoeContiActionNet(BaseNet):

  def __init__(self, feature_dim, action_dim):
    super().__init__()
    self.fc_pi = layer_init(nn.Linear(feature_dim, action_dim), 1e-3)
    self.std = nn.Parameter(torch.zeros((1, action_dim)))

  def forward(self, phi):
    mean = F.tanh(self.fc_pi(phi))
    std = F.softplus(self.std).expand(mean.size(0), -1)

    return {
        'mean': mean,
        'std': std,
    }


class DoeCriticNet(BaseNet):

  def __init__(self, state_dim, num_options, hidden_units=(64, 64),
               gate=F.relu):
    super().__init__()
    dims = (state_dim,) + hidden_units
    self.layers = nn.ModuleList([
        layer_init(nn.Linear(dim_in, dim_out))
        for dim_in, dim_out in zip(dims[:-1], dims[1:])
    ])
    self.gate = gate
    self.feature_dim = dims[-1]
    self.logits_lc = layer_init(nn.Linear(dims[-1], num_options))

  def forward(self, x):
    for layer in self.layers:
      x = self.gate(layer(x))
    x = self.logits_lc(x)
    return x


class DoeContiOneOptionNet(BaseNet):

  def __init__(self,
               state_dim,
               action_dim,
               num_options,
               nhead=4,
               dmodel=40,
               nlayers=3,
               nhid=50,
               dropout=0.2):
    '''
    nhead: number of heads for multiheadattention
    dmodel: embedding size & transormer input size (both decoder & encoder)
            Must Divisible by nhead
    nhid: hidden dimension for transformer
    dropout: dropout for transformer
    '''
    super().__init__()
    ## transformer
    ## encoder
    # option embedding
    self.embed_option = nn.Embedding(num_options, dmodel)
    ## decoder
    # norm state, option concatenation
    self.de_concat_norm = nn.LayerNorm(state_dim + dmodel)
    # todo: should use option embed vector (encoder embedding)
    # or embed_option embedding (separate decoder embedding)?
    # map state, option concatenation -> dmodel
    self.de_so_lc = layer_init(nn.Linear(state_dim + dmodel, dmodel))
    self.de_logtis_lc = layer_init(nn.Linear(dmodel, num_options))

    self.doe = nn.Transformer(dmodel, nhead, nlayers, nlayers, nhid, dropout)

    ## Primary Action
    self.action_nets = nn.ModuleList([
        DoeContiActionNet(state_dim + dmodel, action_dim)
        for _ in range(num_options)
    ])
    self.act_obs_norm = nn.LayerNorm(state_dim + dmodel)

    ## Critic Nets
    self.q_o_st = DoeCriticNet(state_dim + dmodel, num_options)

    self.num_options = num_options
    self.action_dim = action_dim
    self.to(Config.DEVICE)

  def forward(self, obs, prev_options):
    '''
    num_workers: batch_size
    Assumptions:
        obs: config.state_normalizer = MeanStdNormalizer()
    Params:
        obs: [num_workers, state_dim]
        prev_options: [num_workers, 1]

    Returns:
        pot: [num_workers, num_options]
        log_pot: [num_workers, num_options]
        ot: [num_workers]
        q_o_st: [num_workers, num_options]
        pat_mean: [num_workers, act_dim]
        pat_std: [num_workers, act_dim]
    '''
    obs = tensor(obs)

    ## beginning of options part: transformer forward
    # encoder inputs
    num_workers = obs.shape[0]
    # embed_all_idx: [num_options, num_workers]
    embed_all_idx = range_tensor(self.num_options).repeat(num_workers, 1).t()
    # wt: [num_options, num_workers, dmodel(embedding size in init)]
    wt = self.embed_option(embed_all_idx)

    # decoder inputs
    # vt_1: v_{t-1} [1, num_workers, dmodel(embedding size in init)]
    vt_1 = self.embed_option(prev_options.t())
    # obs_cat_1: \tilde{S}_{t-1} [1, num_workers, state_dim + dmodel]
    obs_cat_1 = torch.cat([obs.unsqueeze(0), vt_1], dim=-1)
    obs_cat_1 = self.de_concat_norm(obs_cat_1)
    # obs_hat_1: \tilde{S}_{t-1} [1, num_workers, dmodel]
    obs_hat_1 = self.de_so_lc(obs_cat_1)

    # transformer outputs
    # dt: [1, num_workers, dmodel]
    dt = self.doe(wt, obs_hat_1)
    # pot_logits: [1, num_workers, num_options]
    pot_logits = self.de_logtis_lc(dt)
    # pot_logits/pot/log_pot: [num_workers, num_options]
    pot_logits = pot_logits.squeeze(0)
    pot = F.softmax(pot_logits, dim=-1)
    log_pot = F.log_softmax(pot_logits, dim=-1)

    ## sample options
    pot_dist = torch.distributions.Categorical(probs=pot)
    # ot: [num_workers]
    ot = pot_dist.sample()

    ## beginning of actions part
    # vt: v_t [1, num_workers, dmodel(embedding size in init)]
    vt = self.embed_option(ot.unsqueeze(0))
    # obs_cat: [1, num_workers, state_dim + dmodel(embedding size in init)]
    obs_cat = torch.cat([obs.unsqueeze(0), vt], dim=-1)
    # obs_hat: \tilde{S}_t [1, num_workers, dmodel]
    obs_hat = self.act_obs_norm(obs_cat)

    # generate batch inputs for each option
    batch_idx = range_tensor(num_workers)
    # pat_mean/pat_std: [num_workers, act_dim]
    pat_mean = tensor(np.zeros([num_workers, self.action_dim]))
    pat_std = tensor(np.zeros([num_workers, self.action_dim]))
    for o in range(self.num_options):
      mask = ot == o
      if mask.any():
        obs_hat_o = obs_hat.squeeze(0)[mask, :]
        pat_o = self.action_nets[o](obs_hat_o)
        pat_mean[mask] = pat_o['mean']
        pat_std[mask] = pat_o['std']

    q_o_st = self.q_o_st(obs_hat.squeeze(0))

    return {
        'pot': pot,
        'log_pot': log_pot,
        'ot': ot,
        'pot_dist': pot_dist,
        'q_o_st': q_o_st,
        'pat_mean': pat_mean,
        'pat_std': pat_std,
    }
