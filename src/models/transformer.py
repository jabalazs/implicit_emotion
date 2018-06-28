"""
Code copied from http://nlp.seas.harvard.edu/2018/04/03/attention.html
almost in its entirety with minor modifications
"""

import math
import copy
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
# import matplotlib.pyplot as plt
# import seaborn

import colored_traceback
colored_traceback.add_hook(always=True)

from ..optim.optim import NoamOpt

np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)


class EncoderDecoder(nn.Module):
    """
    Standard high-level encoder-decoder architecture
    """
    def __init__(self, encoder, decoder, src_embed, tgt_embed, generator):
        super(EncoderDecoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.generator = generator

    def forward(self, src, tgt, src_mask, tgt_mask):
        encoded = self.encode(src, src_mask)
        return self.decode(encoded, src_mask, tgt, tgt_mask)

    def encode(self, src, src_mask):
        return self.encoder(self.src_embed(src), src_mask)

    def decode(self, memory, src_mask, tgt, tgt_mask):
        # memory here is just the output of the encoder at the sequence level,
        # or in other words the tensor containing the vector representation for
        # each word; dim = (batch_size, sequence_length, hidden_dim)
        return self.decoder(self.tgt_embed(tgt), memory, src_mask, tgt_mask)


class Generator(nn.Module):
    """
    Generate a logprob distribution over the vocab conditioned on a vector

    This layer is composed of a simple Linear projection layer and a
    log_softmax function for obtaining the logprobs
    """

    def __init__(self, d_model, vocab_size):
        super(Generator, self).__init__()
        self.proj = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        """Generate a logprob distribution based on word-level representations

        Parameters
        ----------

        x: torch.FloatTensor of dim ()"""
        return F.log_softmax(self.proj(x), dim=-1)


def clones(module, N):
    """Create N identical copies of module"""
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


class LayerNorm(nn.Module):
    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        std = x.std(dim=-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2


class SublayerConnection(nn.Module):
    """
    A residual connection followed by a layer norm
    """
    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        return x + self.dropout(sublayer(self.norm(x)))


class Encoder(nn.Module):
    """Core encoder is a stack of num_layers encoder layers"""
    def __init__(self, layer, num_layers):
        super(Encoder, self).__init__()
        self.layers = clones(layer, num_layers)
        # NOTE: `size` here refers to a custom property of our layers, not to
        # the built-in tensor.size() function
        self.norm = LayerNorm(layer.size)

    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)


class EncoderLayer(nn.Module):
    """One encoder layer is composed of self attention and feed forward"""
    def __init__(self, size, attention, feed_forward, dropout):
        super(EncoderLayer, self).__init__()
        self.attention = attention
        self.feed_forward = feed_forward
        self.sublayers = clones(SublayerConnection(size, dropout), 2)
        self.size = size

    def forward(self, x, mask):
        """mask: (batch_size, 1, seq_len)"""
        x = self.sublayers[0](x, lambda x: self.attention(x, x, x, mask))
        return self.sublayers[1](x, self.feed_forward)


class Decoder(nn.Module):
    """Generic decoder composed of num_layers decoder layers with masking"""
    def __init__(self, layer, num_layers):
        super(Decoder, self).__init__()
        self.layers = clones(layer, num_layers)
        self.norm = LayerNorm(layer.size)

    def forward(self, x, memory, src_mask, tgt_mask):
        """Here the input x is the embedded target sequence, and memory is the
        output of the encoder"""
        for layer in self.layers:
            x = layer(x, memory, src_mask, tgt_mask)
        return self.norm(x)


class DecoderLayer(nn.Module):
    def __init__(self, size, slf_attn, src_attn, feed_forward, dropout):
        super(DecoderLayer, self).__init__()
        self.size = size
        self.slf_attn = slf_attn
        self.src_attn = src_attn
        self.feed_forward = feed_forward
        self.sublayers = clones(SublayerConnection(size, dropout), 3)

    def forward(self, x, memory, src_mask, tgt_mask):
        m = memory
        x = self.sublayers[0](x, lambda x: self.slf_attn(x, x, x, tgt_mask))
        x = self.sublayers[1](x, lambda x: self.src_attn(x, m, m, src_mask))
        return self.sublayers[2](x, self.feed_forward)


def attention(query, key, value, mask=None, dropout=None):
    """
    Parameters
    ----------

    query: torch.FloatTensor, dimension(Batch, num_heads, seq_len, d_k)
    key: torch.FloatTensor, dimension(Batch, num_heads, seq_len, d_k)
    value: torch.FloatTensor, dimension(Batch, num_heads, seq_len, d_k)
    """
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    p_attn = F.softmax(scores, dim=-1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn


class MultiHeadedAttention(nn.Module):
    def __init__(self, num_heads, d_model, dropout=0.1):
        super(MultiHeadedAttention, self).__init__()
        assert d_model % num_heads == 0

        # We'll have num_heads heads of dimension d_k
        self.d_k = d_model // num_heads

        self.num_heads = num_heads
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):
        if mask is not None:
            # (batch_size, 1, seq_len) -> (batch_size, 1, 1, seq_len)
            mask = mask.unsqueeze(1)
        batch_size = query.size(0)

        # Do all the linear projs in batch from d_model -> num_heads x d_k
        query, key, value = \
            [l(x).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
             for l, x in zip(self.linears, (query, key, value))]

        # Apply attention on all the projected vectors in batch
        x, self.attn = attention(query, key, value, mask=mask,
                                 dropout=self.dropout)

        # Concat using a view and apply final layer
        # Q: why are we transposing dimensions 1 and 2 here?
        # A: Because the attention returns a tensor of shape
        # (batch_size, h, seq_len, d_k) and we want to make it
        # (batch_size, seq_len, h, d_k) so we can later recover
        # d_model = num_heads * d_k
        x = (x.transpose(1, 2).contiguous()
             .view(batch_size, -1, self.num_heads * self.d_k))

        # Q: We already iterated over all layers, why are we using the last
        # layer again?
        # A: We haven't iterated over all layers. self.linears has 4 elements;
        # when doing the zip operations only the first 3 are matched with the
        # (query, key, value) tuple.
        return self.linears[-1](x)


class PositionwiseFeedforward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedforward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(F.relu(self.w_1(x))))


class Embeddings(nn.Module):
    def __init__(self, vocab_size, d_model):
        super(Embeddings, self).__init__()
        self.lut = nn.Embedding(vocab_size, d_model)
        self.d_model = d_model

    def forward(self, x):
        return self.lut(x) * math.sqrt(self.d_model)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Compute the positional enconding once in log space
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) *
                             -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # NOTE: in the original implementation this is declared as a Variable
        # with `requires_grad=False`
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


class LabelSmoothing(nn.Module):

    """Implement label smoothing"""

    def __init__(self, num_labels, padding_idx=None, smoothing=0.0):
        """

        Parameters
        ----------
        num_labels : int
        padding_idx : TODO, optional
        smoothing : TODO, optional


        """
        super(LabelSmoothing, self).__init__()

        self.num_labels = num_labels
        self.padding_idx = padding_idx
        self.smoothing = smoothing
        self.criterion = nn.KLDivLoss(size_average=False)
        self.confidence = 1.0 - smoothing
        self.true_dist = None

    def forward(self, x, target):
        """

        Parameters
        ----------
        x : torch.FLoatTensor (batch, num_classes)
            Log probabilities outputted by prediction model
        target : torch.LongTensor (batch)
            True labels. Each element should be in [0, num_classes - 1]

        Returns
        -------
        TODO

        """
        assert x.size(1) == self.num_labels
        true_dist = x.clone()
        true_dist = true_dist.detach()
        true_dist.fill_(self.smoothing / (self.num_labels - 2))
        true_dist.scatter_(1, target.unsqueeze(1), self.confidence)
        if self.padding_idx is not None:
            true_dist[:, self.padding_idx] = 0
            mask = torch.nonzero(target == self.padding_idx)
            if mask.nelement() > 0:
                true_dist.index_fill_(0, mask.squeeze(), 0.0)
        self.true_dist = true_dist
        return self.criterion(x, true_dist)


def subsequent_mask(size):
    attn_shape = (1, size, size)
    subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
    return torch.from_numpy(subsequent_mask) == 0


class Batch:
    def __init__(self, src, tgt=None, pad=0):
        self.src = src
        self.src_mask = (src != pad).unsqueeze(-2)
        if tgt is not None:
            self.tgt = tgt[:, :-1]
            self.tgt_y = tgt[:, 1:]
            self.tgt_mask = self.make_std_mask(self.tgt, pad)
            self.num_tokens = (self.tgt_y != pad).sum().item()

    @staticmethod
    def make_std_mask(tgt, pad):
        tgt_mask = (tgt != pad).unsqueeze(-2)
        tgt_mask = (tgt_mask &
                    subsequent_mask(tgt.size(-1)).type_as(tgt_mask))
        return tgt_mask


def data_gen(vocab_size, batch_size, num_batches):
    """Generate random data for a src-tgt copy task

    Parameters
    ----------
    vocab_size : TODO
    batch_size : TODO
    num_batches : TODO

    Returns
    -------
    TODO

    """
    batches = []
    for i in range(num_batches):
        data = torch.from_numpy(np.random.randint(1, vocab_size,
                                                  size=(batch_size, 10)))
        data = data.cuda()
        data[:, 0] = 1
        src = data
        tgt = data
        batches.append(Batch(src, tgt, 0))
    return batches


class SimpleLossCompute(object):

    """A simple loss compute and train function"""

    def __init__(self, generator, criterion, opt=None):
        """TODO: to be defined.

        Parameters
        ----------
        generator : TODO
        criterion : TODO
        opt : TODO, optional


        """
        self.generator = generator
        self.criterion = criterion
        self.opt = opt

    def __call__(self, x, y, norm):
        """
        Parameters
        ----------
        x: torch.FloatTensor, dim=(batch_size, seq_len, model_dim)
            A tensor containing word level representations
        """
        x = self.generator(x)
        try:
            loss = self.criterion(x.contiguous().view(-1, x.size(-1)),
                                  y.contiguous().view(-1))
            norm_loss = loss / norm
        except RuntimeError:
            import ipdb; ipdb.set_trace(context=10)

        norm_loss.backward()
        if self.opt is not None:
            self.opt.step()
            self.opt.zero_grad()
        return norm_loss.item() * norm


def make_model(src_vocab_size, tgt_vocab_size, num_layers=6,
               d_model=512, d_ff=2048, num_heads=8, dropout=0.1):
    """Build model from hyperparameters

    Parameters
    ----------
    src_vocab_size: int
        Size of the source vocabulary
    tgt_vocab_size: int
        Size of the target vocabulary
    num_layers: int
        Number of encoder and decoder layers
    d_model: int
        Model dimensionality
    d_ff: int
        Position-wise Feed-Forward Network dimensionality
    num_heads: int
        The number of heads for the Multi-Head Attention module
    dropout: float
    """
    c = copy.deepcopy
    attn = MultiHeadedAttention(num_heads, d_model)
    ff = PositionwiseFeedforward(d_model, d_ff, dropout)
    position = PositionalEncoding(d_model, dropout)
    model = EncoderDecoder(
        Encoder(EncoderLayer(d_model, c(attn), c(ff), dropout), num_layers),
        Decoder(DecoderLayer(d_model, c(attn), c(attn), c(ff), dropout), num_layers),
        nn.Sequential(Embeddings(src_vocab_size, d_model), c(position)),
        nn.Sequential(Embeddings(tgt_vocab_size, d_model), c(position)),
        Generator(d_model, tgt_vocab_size)
    )

    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    return model


def run_epoch(data_iter, model, loss_compute):
    start = time.time()
    total_tokens = 0
    total_loss = 0
    tokens = 0
    for i, batch in enumerate(data_iter):
        out = model.forward(batch.src, batch.tgt,
                            batch.src_mask, batch.tgt_mask)
        loss = loss_compute(out, batch.tgt_y, batch.num_tokens)
        total_loss += loss
        total_tokens += batch.num_tokens
        tokens += batch.num_tokens
        if i % 50 == 1:
            elapsed = time.time() - start
            print(f'Epoch Step: {i}, Loss: {loss / batch.num_tokens}, '
                  f'Tokens per sec: {tokens / elapsed}')
            start = time.time()
            tokens = 0

    return total_loss / total_tokens


global max_src_in_batch, max_tgt_in_batch


def batch_size_fn(new, count, sofar):
    """Keep augmenting batch and calculate total number of tokens + padding"""
    global max_src_in_batch, max_tgt_in_batch
    if count == 1:
        max_src_in_batch = 0
        max_tgt_in_batch = 0

    max_src_in_batch = max(max_src_in_batch, len(new.src))
    max_tgt_in_batch = max(max_tgt_in_batch, len(new.tgt) + 2)
    src_elements = count * max_src_in_batch
    tgt_elements = count * max_tgt_in_batch
    return max(src_elements, tgt_elements)


def get_std_opt(model):
    model_size = model.src_embed[0].d_model
    factor = 2
    warmup = 4000
    optimizer = torch.optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98),
                                 eps=1e-9)

    return NoamOpt(model_size, factor, warmup, optimizer)


def predict_synthetic_data():
    vocab_size = 11
    criterion = LabelSmoothing(num_labels=vocab_size, padding_idx=0,
                               smoothing=0.0)
    model = make_model(vocab_size, vocab_size, num_layers=2)
    model = model.cuda()
    # model_opt = NoamOpt(
    #     model.src_embed[0].d_model,
    #     factor=1,
    #     warmup=4000,
    #     optimizer=torch.optim.Adam(
    #         model.parameters(),
    #         lr=0,
    #         betas=(0.9, 0.98),
    #         eps=1e-9
    #     )
    # )
    model_opt = torch.optim.Adam(
            model.parameters(),
            lr=0.0001,
            # betas=(0.9, 0.98),
            # eps=1e-9
        )

    batches = data_gen(vocab_size, 30, 20)
    for epoch in range(1000):
        model.train()
        run_epoch(batches, model,
                  SimpleLossCompute(model.generator, criterion, model_opt))
        # model.eval()
        # run_epoch(data_gen(vocab_size, 30, 5), model,
        #           SimpleLossCompute(model.generator, criterion, None))


if __name__ == '__main__':
    predict_synthetic_data()
    exit()
    # plt.figure(figsize=(15, 5))
    # pe = PositionalEncoding(20, 0)
    # y = pe.forward(torch.zeros(1, 100, 20))
    # plt.plot(np.arange(100), y[0, :, 4:8].numpy())
    # plt.legend([f'dim {p}' for p in [4, 5, 6, 7]])
    # plt.show()

    # tmp_model = make_model(10, 10, 2)
    # print(tmp_model)

    # opts = [NoamOpt(512, 1, 4000, None),
    #         NoamOpt(512, 1, 8000, None),
    #         NoamOpt(256, 1, 4000, None)]
    # plt.plot(np.arange(1, 20000), [[opt.rate(i) for opt in opts] for i in range(1, 20000)])
    # plt.legend(['512:4000', '512:8000', '256:4000'])
    # plt.show()

    # crit = LabelSmoothing(5, 1, 0.4)
    # crit = LabelSmoothing(5, padding_idx=1, smoothing=0.4)
    # predict = torch.FloatTensor([[0, 0.2, 0.7, 0.1, 0],
    #                              [0, 0.2, 0.7, 0.1, 0],
    #                              [0, 0.2, 0.7, 0.1, 0]])
    # v = crit(predict.log(),
    #          torch.LongTensor([2, 1, 0]))

    # crit = LabelSmoothing(5, padding_idx=0, smoothing=0.1)

    # def loss(x):
    #     d = x + 3 * 1
    #     predict = torch.FloatTensor([[0, x / d, 1 / d, 1 / d, 1 / d],
    #                                  ])
    #     kldiv_loss = crit(predict.log(), torch.LongTensor([1]))[0]
    #     return kldiv_loss

    # plt.plot(np.arange(1, 100), [loss(x) for x in range(1, 100)])
    # plt.show()

    crit = LabelSmoothing(5, 0, 0.4)
    predict = torch.FloatTensor([[0, 0.2, 0.7, 0.1, 0],
                                 [0, 0.2, 0.7, 0.1, 0],
                                 [0, 0.2, 0.7, 0.1, 0]])
    v = crit(predict.log(),
             torch.LongTensor([2, 1, 0]))

    # Show the target distributions expected by the system.
    # plt.imshow(crit.true_dist)
    crit = LabelSmoothing(5, padding_idx=0, smoothing=0.1)

    def loss(x):
        d = x + 3 * 1
        predict = torch.FloatTensor([
            [0, x / d, 1 / d, 1 / d, 1 / d],
            # [x, x, x, x, x],
            # [1, 1, 1, 1, 1],
        ])
        # print(predict)
        kl_divergence = crit(predict.log(), torch.LongTensor([1])).item()
        return kl_divergence

    # plt.plot(np.arange(1, 100), [loss(x) for x in range(1, 100)])
    # plt.show()
