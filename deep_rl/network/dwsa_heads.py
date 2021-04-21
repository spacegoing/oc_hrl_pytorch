#######################################################################
# Copyright (C) 2017 Shangtong Zhang(zhangshangtong.cpp@gmail.com)    #
# Permission given to modify the code as long as you keep this        #
# declaration at the top                                              #
#######################################################################
from .network_utils import *
from .network_bodies import *
import math

debug_flag = False


class LearnablePositionalEncoding(nn.Module):

  def __init__(self, dmodel, max_len=5000, dropout=0.1):
    super().__init__()
    self.dropout = nn.Dropout(p=dropout)
    self.pe = tensor(nn.Parameter(torch.zeros(max_len, 1, dmodel)))
    nn.init.normal_(self.pe)
    # todo: nn.init.orthogonal_(self.pos_encoder.weight)

  def forward(self, x):
    x = x + self.pe[:x.size(0), :]
    return self.dropout(x)


class SkillEncoder(nn.Module):

  def __init__(self, dmodel, nhead, nhid, nlayers, max_seq_len, dropout=0):
    super().__init__()
    self.dmodel = dmodel
    self.nhead = nhead
    # todo: hyper_param  positional_dropout does pos_encoder need dropout?
    self.pos_encoder = LearnablePositionalEncoding(
        dmodel, max_seq_len, dropout=0.1)

    # transformer encoder
    encoder_layers = nn.TransformerEncoderLayer(dmodel, nhead, nhid, dropout)
    encoder_norm = nn.LayerNorm(dmodel)
    self.transformer_encoder = nn.TransformerEncoder(encoder_layers, nlayers,
                                                     encoder_norm)
    self.init_weights()

  def _generate_square_subsequent_mask(self, sz):
    mask = tensor(torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(
        mask == 1, float(0.0))
    return mask

  def init_weights(self):
    """Initiate parameters in the transformer model."""
    for p in self.parameters():
      if p.dim() > 1:
        nn.init.xavier_uniform_(p)

  def forward(self, src, src_key_padding_mask):
    src_seq_len = src.shape[0]
    src_mask = self._generate_square_subsequent_mask(src_seq_len)

    src = src * math.sqrt(self.dmodel)
    src = self.pos_encoder(src)
    memory = self.transformer_encoder(src, src_mask, src_key_padding_mask)

    return memory


class SkillDecoderLayer(nn.Module):

  def __init__(self, d_model, nhead, nhid, dropout=0.1):
    super().__init__()
    self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
    # Implementation of Feedforward model
    self.linear1 = nn.Linear(d_model, nhid)
    self.linear2 = nn.Linear(nhid, d_model)

    self.norm1 = nn.LayerNorm(d_model)
    self.norm2 = nn.LayerNorm(d_model)
    self.dropout = nn.Dropout(dropout)
    self.dropout1 = nn.Dropout(dropout)
    self.dropout2 = nn.Dropout(dropout)

  def forward(self, tgt, memory, memory_key_padding_mask, **kwargs):
    """Pass the inputs (and mask) through the decoder layer.
    No self attention, mask on tgt, no mask on memory, only padding_mask
      Args:
          tgt: the sequence to the decoder layer (required).
          memory: the sequence from the last layer of the encoder (required).
          memory_key_padding_mask: the mask for the memory keys per batch (optional).
      Shape:
          see the docs in Transformer class.
    """
    tgt2 = self.multihead_attn(
        tgt, memory, memory, key_padding_mask=memory_key_padding_mask)[0]
    tgt = tgt + self.dropout(tgt2)
    tgt = self.norm1(tgt)
    tgt2 = self.linear2(self.dropout1(F.relu(self.linear1(tgt))))
    tgt = tgt + self.dropout2(tgt2)
    tgt = self.norm2(tgt)
    return tgt


class SkillDecoder(nn.Module):

  def __init__(self, dmodel, nhead, nhid, nlayers, dropout=0):
    super().__init__()
    self.dmodel = dmodel
    self.nhead = nhead

    # transformer decoder
    decoder_layer = SkillDecoderLayer(dmodel, nhead, nhid, dropout)
    decoder_norm = nn.LayerNorm(dmodel)
    self.transformer_decoder = nn.TransformerDecoder(decoder_layer, nlayers,
                                                     decoder_norm)

    self.init_weights()

  def init_weights(self):
    """Initiate parameters in the transformer model."""
    for p in self.parameters():
      if p.dim() > 1:
        nn.init.xavier_uniform_(p)

  def forward(self, tgt, memory, memory_key_padding_mask):
    """Pass the inputs (and mask) through the decoder layer.
    No self attention, mask on tgt, no mask on memory, only padding_mask
      Args:
          tgt: the sequence to the decoder layer (required).
          memory: the sequence from the last layer of the encoder (required).
          memory_key_padding_mask: the mask for the memory keys per batch (optional).
      Shape:
          see the docs in Transformer class.
    """
    tgt = tgt * math.sqrt(self.dmodel)
    output = self.transformer_decoder(
        tgt, memory, memory_key_padding_mask=memory_key_padding_mask)
    return output


class WsaFFN(BaseNet):

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


class WsaActionNet(BaseNet):

  def __init__(self, concat_dim, action_dim, hidden_units=(64, 64)):
    super().__init__()
    self.decoder = WsaFFN(concat_dim, hidden_units)
    self.mean_fc = layer_init(nn.Linear(self.decoder.out_dim, action_dim), 1e-3)
    self.std_fc = layer_init(nn.Linear(self.decoder.out_dim, action_dim), 1e-3)

  def forward(self, obs):
    # obs: [num_workers, dmodel+state_dim]
    out = self.decoder(obs)
    # obs: [num_workers, hidden_units[-1]]
    mean = torch.tanh(self.mean_fc(out))
    std = F.softplus(self.std_fc(out))
    return mean, std


class WsaNet(BaseNet):

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

    # skill embedding
    num_embed = num_options + 1  # one extra for padding
    self.embed_option = nn.Embedding(num_embed, dmodel)
    nn.init.orthogonal_(self.embed_option.weight)

    self.skill_encoder = SkillEncoder(
        dmodel, nhead, nhid, nlayers, config.rollout_length, dropout=0)

    ## Skill policy
    # init prob P_0(O|S)
    self.init_po_ffn = WsaFFN(state_dim, hidden_units=(64, num_options))

    #w decoder P(O_t|S_t,O_{t-1...k})
    # todo: other implementation?
    self.skill_decoder_lc = layer_init(nn.Linear(dmodel, 1))
    self.de_state_lc = layer_init(nn.Linear(state_dim, dmodel))
    self.de_state_norm = nn.LayerNorm(dmodel)
    self.skill_decoder = SkillDecoder(dmodel, nhead, nhid, nlayers, dropout=0)

    ## Primary Action
    # todo: no concat but +;
    concat_dim = state_dim + dmodel
    self.act_concat_norm = nn.LayerNorm(concat_dim)
    self.single_transformer_action_net = config.single_transformer_action_net
    self.act_decoder = WsaActionNet(
        concat_dim, action_dim, hidden_units=config.hidden_units)

    ## Critic Nets
    # todo: whether share encoder with skill?
    # skill Q-value embedding
    # num_embed = num_options + 1  # one extra for padding
    # self.embed_qso = nn.Embedding(num_embed, dmodel)
    # nn.init.orthogonal_(self.embed_qso.weight)
    # self.qso_encoder = SkillEncoder(
    #     dmodel, nhead, nhid, nlayers, config.rollout_length + 1, dropout=0)
    self.qso_lc = layer_init(nn.Linear(dmodel, 1))
    # self.qso_lc = WsaFFN(dmodel, hidden_units=(64, 64, 1))

    self.vso_lc = layer_init(nn.Linear(dmodel, 1))
    self.vso_lc1 = layer_init(nn.Linear(num_options, 1))

    self.num_options = num_options
    self.action_dim = action_dim
    self.to(Config.DEVICE)

  def forward(self,
              obs,
              skill_lag_mat,
              initial_state_flags,
              task_switch_flag=False):
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
        skill_lag_mat: [num_workers, config.skill_lag].
                       self.config.padding_mask_token for padding positions
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
    # Preprocess skill_lag_mat
    skill_lag_mat = torch.from_numpy(skill_lag_mat)
    skill_padding_mask = skill_lag_mat.eq(self.config.padding_mask_token)

    ## beginning of options part: transformer forward
    # encoder inputs
    num_workers = obs.shape[0]
    # embed_all_idx: [num_options, num_workers]
    embed_all_idx = range_tensor(self.num_options).repeat(num_workers, 1).t()
    # wt: [num_options, num_workers, dmodel(embedding size in init)]
    wt = self.embed_option(embed_all_idx)

    # decoder inputs
    #w ot_1k: {o_{t-1},...,o_{t-k}} [num_workers, config.skill_lag, dmodel(embedding size in init)]
    ot_1k = self.embed_option(skill_lag_mat).detach()
    # ot_1k_tilde: [skill_lag, num_workers, dmodel] = [S, N, E] in pytorch
    ot_1k_tilde = self.skill_encoder(ot_1k.transpose(0, 1), skill_padding_mask)

    #w st_tilde: [num_workers, dmodel]
    st_tilde = F.relu(self.de_state_lc(obs))
    st_tilde = self.de_state_norm(st_tilde)

    #w ot_1k_tilde = ot_1k_tilde + st_tilde
    ot_1k_tilde = ot_1k_tilde + st_tilde.unsqueeze(0)

    # transformer outputs

    #w
    # ot_tilde: [num_o, num_workers, dmodel] = [T, N, E] in pytorch
    ot_tilde = self.skill_decoder(wt, ot_1k_tilde, skill_padding_mask)
    # po_t_logits: [num_workers, num_o] = [N, T, 1] in pytorch
    po_t_logits = self.skill_decoder_lc(ot_tilde.transpose(0, 1)).squeeze(-1)

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
    obs_hat_a = obs_cat

    # generate batch inputs for each option
    pat_mean, pat_std = self.act_decoder(obs_hat_a)

    ## beginning of value fn
    # # FFN Version
    # # obs_hat: [num_workers, state_dim + dmodel]
    # obs_cat = torch.cat([obs, ot], dim=-1)
    # # obs_hat: [num_workers, state_dim + dmodel]
    # obs_hat = self.q_concat_norm(obs_cat)
    # q_o_st = self.q_o_st(obs_hat)
    # # Add delib cost
    # delib_cost = torch.zeros_like(q_o_st)
    # delib_cost[range_tensor(q_o_st.shape[0]),
    #            prev_options
    #            .squeeze(-1)] -= self.config.delib * torch.abs(q_o_st).mean()
    # q_o_st = q_o_st + delib_cost
    if debug_flag == True:
      if not skill_padding_mask[:, -1].all():
        import ipdb
        ipdb.set_trace(context=7)
    #w
    # todo: whether share same net?
    # skill_mat = torch.cat([skill_lag_mat, ot_hat.unsqueeze(1)], dim=1)
    # # Fix ot position
    # any_padding_mask = skill_padding_mask.any(-1)
    # for i, t in enumerate(any_padding_mask):
    #   if t:
    #     first_pad_idx = torch.where(skill_padding_mask[i])[0][0]
    #     skill_mat[i, first_pad_idx] = skill_mat[i, -1]
    #     skill_mat[i, -1] = self.config.padding_mask_token
    # vfn_padding_mask = skill_mat.eq(self.config.padding_mask_token)
    # otk = self.embed_option(skill_mat)
    # # otk_tilde: [skill_lag+1, num_workers, dmodel] = [S, N, E] in pytorch
    # otk_tilde = self.qfn_encoder(otk.transpose(0, 1), vfn_padding_mask)

    # todo: detach???
    # todo: FFN?
    # ot_tilde: [num_o, num_workers, dmodel] = [T, N, E] in pytorch
    # q_o_st: [num_workers, num_o] = [N, T, 1] in pytorch
    q_o_st = self.qso_lc(ot_tilde.transpose(0, 1)).squeeze(-1)

    # v_st = (q_o_st * po_t).sum(axis=1).unsqueeze(-1)
    v_st_o = self.vso_lc(ot_tilde.transpose(0, 1)).squeeze(-1)
    v_st = self.vso_lc1(v_st_o)

    return {
        'po_t': po_t,
        'po_t_log': po_t_log,
        'ot': ot_hat.unsqueeze(-1),
        'po_t_dist': po_t_dist,
        'q_o_st': q_o_st,
        'q_ot_st': q_o_st.gather(1, ot_hat.unsqueeze(-1)),
        'v_st': v_st,
        'pat_mean': pat_mean,
        'pat_std': pat_std,
        'wt': self.embed_option(range_tensor(self.num_options)),
    }
