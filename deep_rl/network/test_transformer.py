import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchtext
from torchtext.data.utils import get_tokenizer
from torch.nn import TransformerEncoder, TransformerEncoderLayer, TransformerDecoder, TransformerDecoderLayer, LayerNorm, Transformer

debug_flag = False


class PositionalEncoding(nn.Module):

  def __init__(self, d_model, dropout=0.1, max_len=5000):
    '''
      d_model=20; dropout=0.1; max_len=50
    '''
    super(PositionalEncoding, self).__init__()
    self.dropout = nn.Dropout(p=dropout)

    self.pe = nn.Parameter(torch.zeros(max_len, 1, d_model))
    nn.init.normal_(self.pe)

    # # Old Block ---------------------------
    # self.pe = nn.Parameter(torch.zeros(max_len, 1, d_model))
    # position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
    # div_term = torch.exp(
    #     torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
    # pe[:, 0::2] = torch.sin(position * div_term)
    # pe[:, 1::2] = torch.cos(position * div_term)
    # pe = pe.unsqueeze(0).transpose(0, 1)
    # self.register_buffer('pe', pe)

  def forward(self, x):
    if debug_flag == True: import ipdb; ipdb.set_trace(context=7)
    x = x + self.pe[:x.size(0), :]
    return self.dropout(x)


class TransformerModel(nn.Module):

  def __init__(self, ntoken, ninp, nhead, nhid, nlayers, dropout=0.5):
    super(TransformerModel, self).__init__()
    self.model_type = 'Transformer'
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


TEXT = torchtext.data.Field(
    tokenize=get_tokenizer("basic_english"),
    init_token='<sos>',
    eos_token='<eos>',
    lower=True)
train_txt, val_txt, test_txt = torchtext.datasets.WikiText2.splits(TEXT)
TEXT.build_vocab(train_txt)
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cuda:4")
# device = torch.device("cpu")


def batchify(data, bsz):
  data = TEXT.numericalize([data.examples[0].text])
  # Divide the dataset into bsz parts.
  nbatch = data.size(0) // bsz
  # Trim off any extra elements that wouldn't cleanly fit (remainders).
  data = data.narrow(0, 0, nbatch * bsz)
  # Evenly divide the data across the bsz batches.
  data = data.view(bsz, -1).t().contiguous()
  return data.to(device)


batch_size = 20
eval_batch_size = 10
train_data = batchify(train_txt, batch_size)
val_data = batchify(val_txt, eval_batch_size)
test_data = batchify(test_txt, eval_batch_size)

bptt = 35


def get_batch(source, i):
  seq_len = min(bptt, len(source) - 1 - i)
  data = source[i:i + seq_len]
  target = source[i + 1:i + 1 + seq_len].view(-1)
  return data, target


ntokens = len(TEXT.vocab.stoi)  # the size of vocabulary
emsize = 200  # embedding dimension
nhid = 100  # the dimension of the feedforward network model in nn.TransformerEncoder
nlayers = 2  # the number of nn.TransformerEncoderLayer in nn.TransformerEncoder
nhead = 2  # the number of heads in the multiheadattention models
dropout = 0.2  # the dropout value
model = TransformerModel(ntokens, emsize, nhead, nhid, nlayers,
                         dropout).to(device)

criterion = nn.CrossEntropyLoss()
lr = 5.0  # learning rate
optimizer = torch.optim.SGD(model.parameters(), lr=lr)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1.0, gamma=0.95)

import time


def train():
  model.train()  # Turn on the train mode
  total_loss = 0.
  start_time = time.time()
  ntokens = len(TEXT.vocab.stoi)
  for batch, i in enumerate(range(0, train_data.size(0) - 1, bptt)):
    if debug_flag == True:
      import ipdb
      ipdb.set_trace(context=7)
    data, targets = get_batch(train_data, i)
    optimizer.zero_grad()
    output = model(data, data)
    loss = criterion(output.view(-1, ntokens), targets)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
    optimizer.step()

    total_loss += loss.item()
    log_interval = 200
    if batch % log_interval == 0 and batch > 0:
      cur_loss = total_loss / log_interval
      elapsed = time.time() - start_time
      print('| epoch {:3d} | {:5d}/{:5d} batches | '
            'lr {:02.2f} | ms/batch {:5.2f} | '
            'loss {:5.2f} | ppl {:8.2f}'.format(epoch, batch,
                                                len(train_data) // bptt,
                                                scheduler.get_lr()[0],
                                                elapsed * 1000 / log_interval,
                                                cur_loss, math.exp(cur_loss)))
      total_loss = 0
      start_time = time.time()


def evaluate(eval_model, data_source):
  eval_model.eval()  # Turn on the evaluation mode
  total_loss = 0.
  ntokens = len(TEXT.vocab.stoi)
  with torch.no_grad():
    for i in range(0, data_source.size(0) - 1, bptt):
      data, targets = get_batch(data_source, i)
      output = eval_model(data, data)
      output_flat = output.view(-1, ntokens)
      total_loss += len(data) * criterion(output_flat, targets).item()
  return total_loss / (len(data_source) - 1)


best_val_loss = float("inf")
epochs = 30  # The number of epochs
best_model = None

for epoch in range(1, epochs + 1):
  epoch_start_time = time.time()
  train()
  val_loss = evaluate(model, val_data)
  print('-' * 89)
  print('| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.2f} | '
        'valid ppl {:8.2f}'.format(epoch, (time.time() - epoch_start_time),
                                   val_loss, math.exp(val_loss)))
  print('-' * 89)

  if val_loss < best_val_loss:
    best_val_loss = val_loss
    best_model = model

  scheduler.step()

test_loss = evaluate(best_model, test_data)
print('=' * 89)
print('| End of training | test loss {:5.2f} | test ppl {:8.2f}'.format(
    test_loss, math.exp(test_loss)))
print('=' * 89)
