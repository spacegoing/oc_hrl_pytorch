#######################################################################
# Copyright (C) 2017 Shangtong Zhang(zhangshangtong.cpp@gmail.com)    #
# Permission given to modify the code as long as you keep this        #
# declaration at the top                                              #
#######################################################################
from .network_utils import *
from .network_bodies import *
import math


class PositionalEncoding(nn.Module):

  def __init__(self, dmodel, dropout=0.1, max_len=5000):
    super().__init__()
    self.dropout = nn.Dropout(p=dropout)
    self.pe = nn.Parameter(torch.zeros(max_len, 1, dmodel))
    nn.init.normal_(self.pe)

  def forward(self, x):
    x = x + self.pe[:x.size(0), :]
    return self.dropout(x)


class EncoderModel(nn.Module):

  def __init__(self, ntoken, ninp, nhead, nhid, nlayers, dropout=0.5):
    super().__init__()
    self.ninp = ninp
    self.embed_src = nn.Embedding(ntoken, ninp)
    self.embed_tgt = nn.Embedding(ntoken, ninp)
    self.pos_encoder = PositionalEncoding(ninp, dropout)

    # transformer encoder
    encoder_layers = TransformerEncoderLayer(ninp, nhead, nhid, dropout)
    encoder_norm = LayerNorm(ninp)
    self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers,
                                                  encoder_norm)
    # transformer decoder
    decoder_layers = TransformerDecoderLayer(ninp, nhead, nhid, dropout)
    decoder_norm = LayerNorm(ninp)
    self.transformer_decoder = TransformerDecoder(decoder_layers, nlayers,
                                                  decoder_norm)
    # self.transformer = Transformer(ninp, nhead, nlayers, nlayers, nhid, dropout)
    self.logits = nn.Linear(ninp, ntoken)

    self.init_weights()

  def get_all_masks(self, src_seq_len, tgt_seq_len, device='cpu'):
    src_mask = self._generate_square_subsequent_mask(src_seq_len, device)
    tgt_mask = self._generate_square_subsequent_mask(tgt_seq_len, device)

    if src_seq_len >= tgt_seq_len:
      memory_mask = src_mask[:tgt_seq_len].clone()
    else:
      memory_mask = tgt_mask[:, :src_seq_len].clone()
    return src_mask, tgt_mask, memory_mask

  def _generate_square_subsequent_mask(self, sz, device='cpu'):
    mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(
        mask == 1, float(0.0))
    return mask.to(device)

  def init_weights(self):
    """Initiate parameters in the transformer model."""
    for p in self.parameters():
      if p.dim() > 1:
        nn.init.xavier_uniform_(p)
    # initrange = 0.1
    # self.encoder.weight.data.uniform_(-initrange, initrange)
    # self.decoder.bias.data.zero_()
    # self.decoder.weight.data.uniform_(-initrange, initrange)

  def forward(self, src, tgt):
    src_seq_len = src.shape[0]
    tgt_seq_len = tgt.shape[0]
    device = src.device
    src_mask, tgt_mask, memory_mask = self.get_all_masks(
        src_seq_len, tgt_seq_len, device)

    if debug_flag == True:
      import ipdb
      ipdb.set_trace(context=7)

    es = self.embed_src(src)
    src = self.embed_src(src) * math.sqrt(self.ninp)
    src = self.pos_encoder(src)
    memory = self.transformer_encoder(src, src_mask)
    output = self.logits(memory)

    tgt = self.embed_tgt(tgt) * math.sqrt(self.ninp)
    # encoder, decoder use the same embed matrix
    # tgt = self.embed_src(tgt) * math.sqrt(self.ninp)
    tgt = self.pos_encoder(tgt)
    # output = self.transformer_decoder(tgt, memory, tgt_mask=tgt_mask)
    output = self.transformer_decoder(
        tgt, memory, tgt_mask=tgt_mask, memory_mask=memory_mask)
    # output = output.permute(1, 0, 2)

    # # use pytorch transformer module
    # output = self.transformer(
    #     src, tgt, src_mask=src_mask, memory_mask=memory_mask, tgt_mask=tgt_mask)

    output = self.logits(output)
    return output


class WsaNet(BaseNet):

  def __init__(self,
               state_dim,
               action_dim,
               num_options,
               max_lag=int(5e3),
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
    self.embed_option = nn.Embedding(num_options, dmodel)
    nn.init.orthogonal_(self.embed_option.weight)
    # positional encoding
    self.pos_encoder = PositionalEncoding(dmodel, dropout, max_lag)
    # todo: nn.init.orthogonal_(self.pos_encoder.weight)
    # todo: large dropout

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
    self.v_logtis_lc = layer_init(nn.Linear(2 * dmodel, num_options))

    self.num_options = num_options
    self.action_dim = action_dim
    self.to(Config.DEVICE)

  def forward(self, obs, prev_options, initial_state_flags, options):
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
    obs_hat_a = obs_cat

    # generate batch inputs for each option
    pat_mean, pat_std = self.act_doe(obs_hat_a)

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

    # Attn Version
    wt = self.embed_option(embed_all_idx)
    # obs_cat: \tilde{S}_{t-1} [2, num_workers, dmodel]
    obs_cat = torch.cat([obs_hat.unsqueeze(0), ot.unsqueeze(0)], dim=0)
    # transformer outputs
    # dt: [2, num_workers, dmodel] [0]: mha_st; [1]: mha_ot
    rdt = self.doe(wt, obs_cat)
    # dt: [num_workers, dmodel(st)+dmodel(o_{t-1})]
    dt = torch.cat([rdt[0].squeeze(0), rdt[1].squeeze(0)], dim=-1)
    if dt.dim() < 2:
      dt = dt.unsqueeze(0)
    # q_o_st: [num_workers, num_options]
    # todo: detach value fn from doe head; only train Linear
    # dt = dt.detach()
    q_o_st = self.v_logtis_lc(dt)

    # if task_switch_flag:
    #   po_t = po_t.detach()
    #   po_t_log = po_t_log.detach()
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
