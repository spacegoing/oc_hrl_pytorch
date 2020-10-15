#######################################################################
# Copyright (C) 2017 Shangtong Zhang(zhangshangtong.cpp@gmail.com)    #
# Permission given to modify the code as long as you keep this        #
# declaration at the top                                              #
#######################################################################

from .network_utils import *
from .network_bodies import *
import math


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


class MultiheadAttention(nn.Module):

  def __init__(self, qdim, dmodel, nhead):
    super().__init__()

    self.dmodel = dmodel
    self.h = nhead
    self.d_k = dmodel // nhead
    assert self.d_k * nhead == dmodel

    self.q_linear = layer_init(nn.Linear(qdim, dmodel))
    self.v_linear = layer_init(nn.Linear(dmodel, dmodel))
    self.k_linear = layer_init(nn.Linear(dmodel, dmodel))
    self.out = layer_init(nn.Linear(dmodel, dmodel))

  def forward(self, q, k, v):
    '''
    q: [seq_len, batch_size, query_dim]
    k,v: [seq_len, batch_size, dmodel]

    output: [seq_len, batch_size, dmodel]
    '''
    sl, bs = q.size(0), q.size(1)
    # perform linear operation and split into H heads
    q = self.q_linear(q).view(sl, bs, self.h, self.d_k)
    k = self.k_linear(k).view(-1, bs, self.h, self.d_k)
    v = self.v_linear(v).view(-1, bs, self.h, self.d_k)

    # transpose to get dimensions bs * H * sl * d_model
    q = q.transpose(0, 1).transpose(1, 2)
    k = k.transpose(0, 1).transpose(1, 2)
    v = v.transpose(0, 1).transpose(1, 2)

    # calculate attention using function we will define next

    scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
    scores = F.softmax(scores, dim=-1)
    scores = torch.matmul(scores, v)

    # concatenate heads and put through final linear layer
    concat = scores.transpose(1, 2).transpose(0, 1).contiguous().view(
        sl, bs, self.dmodel)
    output = self.out(concat)

    return output


# class SkillMhaLayer(BaseNet):

#   def __init__(self, state_dim, dmodel, nhead, dim_feedforward=128):
#     super().__init__()
#     self.multihead_attn = MultiheadAttention(state_dim, dmodel, nhead)
#     # Implementation of Feedforward model
#     self.linear1 = layer_init(nn.Linear(dmodel, dim_feedforward))
#     self.linear2 = layer_init(nn.Linear(dim_feedforward, dmodel))

#     self.norm = nn.LayerNorm(dmodel)

#   def forward(self, tgt, memory):
#     tgt2 = self.multihead_attn(tgt, memory, memory)
#     tgt2 = self.linear2(F.relu(self.linear1(tgt2)))
#     tgt2 = self.norm(tgt2)
#     return tgt2

# class SkillPolicy(BaseNet):

#   def __init__(self, num_options, state_dim, embed_dim, nhead, nlayers, nhid):
#     '''
#     hidden_units: layers for FFN(state_dim + embed_dim)
#     state_dim: query dimension (qdim)
#     embed_dim: embedding (skill context vector) dimension (kdim, vdim)
#     '''
#     super().__init__()
#     self.state_mha_layers = nn.ModuleList([
#         SkillMhaLayer(state_dim, embed_dim, nhead, nhid) for i in range(nlayers)
#     ])

#     # FFN(st,ot-1)->ot_logits: concat state_mha and embed_mha
#     self.norm = nn.LayerNorm(embed_dim + embed_dim)
#     dims = (embed_dim + embed_dim,) + (nhid,) + (num_options,)
#     self.ffn = nn.ModuleList([
#         layer_init(nn.Linear(dim_in, dim_out))
#         for dim_in, dim_out in zip(dims[:-1], dims[1:])
#     ])

#     for p in self.parameters():
#       if p.dim() > 1:
#         nn.init.xavier_uniform_(p)

#   def forward(self, state, ot_1, wt):
#     '''
#     wt: memory
#     state: state tgt
#     ot_1: embed tgt
#     '''
#     for mod in self.state_mha_layers:
#       state = mod(state, wt)

#     out = torch.cat([state, ot_1], dim=-1)
#     out = self.norm(out)
#     for layer in self.ffn:
#       out = F.relu(layer(out))
#     return out


class SkillMhaLayer(BaseNet):

  def __init__(self, d_model, nhead, dim_feedforward=128, dropout=0.1):
    super().__init__()
    self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
    # Implementation of Feedforward model
    self.linear1 = nn.Linear(d_model, dim_feedforward)
    self.dropout = nn.Dropout(dropout)
    self.linear2 = nn.Linear(dim_feedforward, d_model)

    self.norm2 = nn.LayerNorm(d_model)
    self.norm3 = nn.LayerNorm(d_model)
    self.dropout2 = nn.Dropout(dropout)
    self.dropout3 = nn.Dropout(dropout)

  def forward(self, tgt, memory):
    tgt2 = self.multihead_attn(tgt, memory, memory)[0]
    tgt = tgt + self.dropout2(tgt2)
    tgt = self.norm2(tgt)
    tgt2 = self.linear2(self.dropout(F.relu(self.linear1(tgt))))
    tgt = tgt + self.dropout3(tgt2)
    tgt = self.norm3(tgt)
    return tgt


class SkillPolicy(BaseNet):

  def __init__(self, dmodel, nhead, nlayers, nhid, dropout):
    super().__init__()
    self.layers = nn.ModuleList(
        [SkillMhaLayer(dmodel, nhead, nhid, dropout) for i in range(nlayers)])
    self.norm = nn.LayerNorm(dmodel)
    for p in self.parameters():
      if p.dim() > 1:
        nn.init.xavier_uniform_(p)

  def forward(self, memory, tgt):
    output = tgt
    for mod in self.layers:
      output = mod(output, memory)
    output = self.norm(output)
    return output


class DoeSkillDecoderNet(BaseNet):

  def __init__(self, dmodel, nhead, nlayers, nhid, dropout=0.0):
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

  def __init__(self, state_dim, hidden_units=(64, 64), gate=F.relu):
    super().__init__()
    dims = (state_dim,) + hidden_units
    self.layers = nn.ModuleList([
        layer_init(nn.Linear(dim_in, dim_out))
        for dim_in, dim_out in zip(dims[:-1], dims[1:])
    ])
    self.gate = gate
    self.out_dim = hidden_units[-1]
    for p in self.layers.parameters():
      if p.dim() > 1:
        nn.init.xavier_uniform_(p)

  def forward(self, obs):
    # obs: [num_workers, dmodel+state_dim]
    for layer in self.layers:
      obs = self.gate(layer(obs))
    return obs


class DoeSingleTransActionNet(BaseNet):

  def __init__(self, concat_dim, action_dim, hidden_units=(64, 64)):
    super().__init__()
    self.decoder = DoeDecoderFFN(concat_dim, hidden_units)
    self.mean_fc = layer_init(nn.Linear(self.decoder.out_dim, action_dim), 1e-3)
    self.std_fc = layer_init(nn.Linear(self.decoder.out_dim, action_dim), 1e-3)

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
    self.logits_lc = layer_init(nn.Linear(self.decoder.out_dim, num_options))

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
               dropout=0.0,
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
    self.config = config

    # option embedding
    self.embed_option = nn.Embedding(num_options, dmodel)
    nn.init.orthogonal_(self.embed_option.weight)

    ## Skill policy
    # init prob P_0(O|S)
    self.init_po_ffn = DoeDecoderFFN(state_dim, hidden_units=(64, num_options))
    # decoder P(O_t|S_t,O_{t-1})
    # self.doe = DoeSkillDecoderNet(dmodel, nhead, nlayers, nhid, dropout)
    # decoder P(O_t|S_t,O_{t-1})
    self.de_state_lc = layer_init(nn.Linear(state_dim, dmodel))
    self.de_state_norm = nn.LayerNorm(dmodel)
    self.de_logtis_lc = layer_init(nn.Linear(2 * dmodel, num_options))
    self.doe = SkillPolicy(dmodel, nhead, nlayers, nhid, dropout)
    # self.skill_policy = SkillPolicy(num_options, state_dim, dmodel, nhead,
    #                                 nlayers, nhid)

    ## Primary Action
    concat_dim = state_dim + dmodel
    self.act_concat_norm = nn.LayerNorm(concat_dim)
    self.single_transformer_action_net = config.single_transformer_action_net
    self.act_doe = DoeSingleTransActionNet(
        concat_dim, action_dim, hidden_units=config.hidden_units)

    ## Critic Nets
    critic_dim = state_dim + dmodel
    self.q_concat_norm = nn.LayerNorm(critic_dim)
    self.q_o_st = DoeCriticNet(critic_dim, num_options, config.hidden_units)

    self.num_options = num_options
    self.action_dim = action_dim
    self.to(Config.DEVICE)

  def forward(self, obs, prev_options, initial_state_flags):
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
        initial_state_flags: [num_workers, 1]

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
    # ot_1: o_{t-1} [1, num_workers, dmodel(embedding size in init)]
    ot_1 = self.embed_option(prev_options.t()).detach()

    # obs_hat: [num_workers, dmodel]
    obs_hat = F.relu(self.de_state_lc(obs))
    obs_hat = self.de_state_norm(obs_hat)
    # obs_cat_1: \tilde{S}_{t-1} [2, num_workers, dmodel]
    obs_cat_1 = torch.cat([obs_hat.unsqueeze(0), ot_1], dim=0)

    # transformer outputs
    # dt: [2, num_workers, dmodel] [0]: mha_st; [1]: mha_ot_1
    rdt = self.doe(wt, obs_cat_1)
    # dt: [num_workers, dmodel(st)+dmodel(o_{t-1})]
    dt = torch.cat([rdt[0].squeeze(0), rdt[1].squeeze(0)], dim=-1)
    if dt.dim() < 2:
      dt = dt.unsqueeze(0)
    # po_t_logits/po_t/po_t_log: [num_workers, num_options]
    po_t_logits = self.de_logtis_lc(dt)
    # po_t_logits = self.skill_policy(obs.unsqueeze(0), ot_1, wt).squeeze(0)

    # handle initial state
    if initial_state_flags.any():
      if initial_state_flags.dim() > 1:
        initial_state_flags = initial_state_flags.squeeze(-1)
      po_t_logits_init = self.init_po_ffn(obs)
      po_t_logits[initial_state_flags] = po_t_logits_init[initial_state_flags]

    po_t = F.softmax(po_t_logits, dim=-1)
    po_t_log = F.log_softmax(po_t_logits, dim=-1)

    ## sample options
    po_t_dist = torch.distributions.Categorical(probs=po_t)
    # ot_hat: [num_workers]
    ot_hat = po_t_dist.sample()

    ## beginning of actions part
    # ot: v_t [num_workers, dmodel(embedding size in init)]
    ot = self.embed_option(ot_hat.unsqueeze(0)).detach().squeeze(0)
    # obs_cat: [num_workers, state_dim + dmodel]
    obs_cat = torch.cat([obs, ot], dim=-1)
    # obs_hat: \tilde{S}_t [1, num_workers, dmodel]
    # obs_hat = self.act_concat_norm(obs_cat)
    obs_hat = obs_cat

    # generate batch inputs for each option
    pat_mean, pat_std = self.act_doe(obs_hat)

    ## beginning of value fn
    # obs_hat: [num_workers, state_dim + dmodel]
    obs_cat = torch.cat([obs, ot], dim=-1)
    # obs_hat: [num_workers, state_dim + dmodel]
    obs_hat = self.q_concat_norm(obs_cat)
    q_o_st = self.q_o_st(obs_hat)
    # Add delib cost
    delib_cost = torch.zeros_like(q_o_st)
    delib_cost[range_tensor(q_o_st.shape[0]),
               prev_options
               .squeeze(-1)] -= self.config.delib * torch.abs(q_o_st).mean()
    q_o_st = q_o_st + delib_cost

    return {
        'po_t': po_t,
        'po_t_log': po_t_log,
        'ot': ot_hat.unsqueeze(-1),
        'po_t_dist': po_t_dist,
        'q_o_st': q_o_st,
        'q_ot_st': q_o_st.gather(1, ot_hat.unsqueeze(-1)),
        'v_st': (q_o_st * po_t).sum(axis=1).unsqueeze(-1),
        'pat_mean': pat_mean,
        'pat_std': pat_std,
        'wt': self.embed_option(range_tensor(self.num_options)),
    }
