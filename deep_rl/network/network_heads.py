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
    mean = torch.tanh(self.fc_pi(phi))
    std = F.softplus(self.std).expand(mean.size(0), -1)

    return {
        'mean': mean,
        'std': std,
    }


class DoeSkillDecoderNet(BaseNet):

  def __init__(self, dmodel, nhead, nlayers, nhid, dropout):
    super().__init__()
    decoder_layers = nn.TransformerDecoderLayer(dmodel, nhead, nhid, dropout)
    decoder_norm = nn.LayerNorm(dmodel)
    self.transformer_decoder = nn.TransformerDecoder(decoder_layers, nlayers,
                                                     decoder_norm)
    for p in self.transformer_decoder.parameters():
      if p.dim() > 1:
        nn.init.xavier_uniform_(p)

  def forward(self, mem, tgt):
    '''
    # mem(wt): [num_options, num_workers, dmodel(embedding size in init)]
    # tgt(obs_hat_1): \tilde{S}_{t-1} [1, num_workers, dmodel]

    # out(dt): [1, num_workers, dmodel]
    '''
    out = self.transformer_decoder(tgt, mem)
    return out


class DoeDecoderFFN(BaseNet):

  def __init__(self, state_dim, hidden_units=(64, 64)):
    super().__init__()
    dims = (state_dim,) + hidden_units
    self.layers = nn.ModuleList([
        layer_init(nn.Linear(dim_in, dim_out))
        for dim_in, dim_out in zip(dims[:-1], dims[1:])
    ])
    for p in self.layers.parameters():
      if p.dim() > 1:
        nn.init.xavier_uniform_(p)

  def forward(self, obs):
    # obs: [num_workers, dmodel+state_dim]
    for layer in self.layers:
      obs = F.relu(layer(obs))
    return obs


class DoeSingleTransActionNet(BaseNet):

  def __init__(self, concat_dim, action_dim, hidden_units=(64, 64)):
    super().__init__()
    self.decoder = DoeDecoderFFN(concat_dim, hidden_units)
    self.mean_fc = layer_init(nn.Linear(hidden_units[-1], action_dim), 1e-3)
    self.std_fc = layer_init(nn.Linear(hidden_units[-1], action_dim), 1e-3)

  def forward(self, obs):
    # obs: [num_workers, dmodel+state_dim]
    out = self.decoder(obs)
    # obs: [num_workers, hidden_units[-1]]
    mean = torch.tanh(self.mean_fc(out))
    std = F.softplus(self.std_fc(out))
    return mean, std


class DoeCriticNet(BaseNet):

  def __init__(self, concat_dim, num_options, hidden_units=(64, 64)):
    super().__init__()
    self.decoder = DoeDecoderFFN(concat_dim, hidden_units)
    self.logits_lc = layer_init(nn.Linear(hidden_units[-1], num_options))

  def forward(self, obs):
    # obs: [num_workers, dmodel]
    out = self.decoder(obs)
    # q_o: [num_workers, num_options]
    q_o = self.logits_lc(out)
    return q_o


class DoeContiOneOptionNet(BaseNet):

  def __init__(self,
               state_dim,
               action_dim,
               num_options,
               nhead=4,
               dmodel=40,
               nlayers=3,
               nhid=50,
               dropout=0.2,
               config=None):
    '''
    nhead: number of heads for multiheadattention
    dmodel: embedding size & transormer input size (both decoder & encoder)
            Must Divisible by nhead
    nhid: hidden dimension for transformer
    dropout: dropout for transformer
    '''
    super().__init__()

    # # test 333
    # dmodel = state_dim

    # option embedding
    self.embed_option = nn.Embedding(num_options, dmodel)

    ## Skill policy: decoder
    self.de_state_lc = layer_init(nn.Linear(state_dim, dmodel))
    self.de_state_norm = nn.LayerNorm(dmodel)
    self.de_logtis_lc = layer_init(nn.Linear(2 * dmodel, num_options))
    self.doe = DoeSkillDecoderNet(dmodel, nhead, nlayers, nhid, dropout)

    ## Primary Action
    concat_dim = state_dim + dmodel
    self.act_concat_norm = nn.LayerNorm(concat_dim)
    self.single_transformer_action_net = config.single_transformer_action_net
    if self.single_transformer_action_net:
      self.act_doe = DoeSingleTransActionNet(
          concat_dim, action_dim, hidden_units=config.hidden_units)
    else:
      self.action_nets = nn.ModuleList(
          [DoeContiActionNet(dmodel, action_dim) for _ in range(num_options)])

    ## Critic Nets
    self.q_concat_norm = nn.LayerNorm(dmodel + dmodel)
    self.q_o_st = DoeCriticNet(dmodel + dmodel, num_options,
                               config.hidden_units)

    self.num_options = num_options
    self.action_dim = action_dim
    self.to(Config.DEVICE)

  def forward(self, obs, prev_options):
    '''
    Naming Conventions:
    1. num_workers: batch_size
    2. if o does not follow timestamp t, it means for all options:
         q_o_st: [num_workers, num_options] $Q_o_t(O,S_t)$
         po_t/po_t_log: [num_workers, num_options] $P(O|S_t,o_{t-1};w_t)$

       if ot, it means for O=ot:
         q_ot_st: [num_workers, 1] $Q_o_t(O=ot, S_t)$
         pot/pot_log: [num_workers, 1] $P(O=ot|S_t,o_{t-1};w_t)$

    Assumptions:
        obs: config.state_normalizer = MeanStdNormalizer()
    Params:
        obs: [num_workers, state_dim]
        prev_options: [num_workers, 1]

    Returns:
        po_t: [num_workers, num_options]
        po_t_log: [num_workers, num_options]
        ot: [num_workers, 1]
            However, ot as intermediate results in this
            function are [num_workers]
        po_t_dist: Categorical(probs: torch.Size([3, 4]))
        q_o_st: [num_workers, num_options]
        q_ot_st: [num_workers, 1]
        v_st: [num_workers, 1]
              V(S_t,O_{t-1}) = \sum_{o\in \O_t} P(o|S_t,O_{t-1})Q(o,S_t)
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
    vt_1 = self.embed_option(prev_options.t()).detach()
    obs_hat = F.relu(self.de_state_lc(obs))
    obs_hat = self.de_state_norm(obs_hat)
    # obs_cat_1: \tilde{S}_{t-1} [2, num_workers, dmodel]
    obs_cat_1 = torch.cat([obs_hat.unsqueeze(0), vt_1], dim=0)

    # transformer outputs
    # dt: [2, num_workers, dmodel]
    dt = self.doe(wt, obs_cat_1)
    # dt: [num_workers, dmodel(state)+dmodel(o_{t-1})]
    dt = torch.cat([dt[0].squeeze(0), dt[1].squeeze(0)], dim=-1)
    # po_t_logits: [num_workers, num_options]
    po_t_logits = self.de_logtis_lc(dt)
    # po_t_logits/po_t/po_t_log: [num_workers, num_options]
    po_t_logits = po_t_logits
    po_t = F.softmax(po_t_logits, dim=-1)
    po_t_log = F.log_softmax(po_t_logits, dim=-1)

    ## sample options
    po_t_dist = torch.distributions.Categorical(probs=po_t)
    # ot: [num_workers]
    ot = po_t_dist.sample()

    ## beginning of actions part
    # vt: v_t [1, num_workers, dmodel(embedding size in init)]
    vt = self.embed_option(ot.unsqueeze(0)).detach().squeeze(0)
    # obs_cat: [num_workers, state_dim + dmodel]
    obs_cat = torch.cat([obs, vt], dim=-1)
    # obs_hat: \tilde{S}_t [1, num_workers, dmodel]
    obs_hat = self.act_concat_norm(obs_cat)

    # generate batch inputs for each option
    if self.single_transformer_action_net:
      pat_mean, pat_std = self.act_doe(obs_hat)
    else:
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

    ## beginning of value fn
    # obs_hat: \tilde{S}_t [1, num_workers, dmodel]
    obs_hat = self.q_concat_norm(dt)
    q_o_st = self.q_o_st(obs_hat)

    return {
        'po_t': po_t,
        'po_t_log': po_t_log,
        'ot': ot.unsqueeze(-1),
        'po_t_dist': po_t_dist,
        'q_o_st': q_o_st,
        'q_ot_st': q_o_st.gather(1, ot.unsqueeze(-1)),
        'v_st': (q_o_st * po_t).sum(axis=1).unsqueeze(-1),
        'pat_mean': pat_mean,
        'pat_std': pat_std,
        'wt': self.embed_option(range_tensor(self.num_options)),
    }
