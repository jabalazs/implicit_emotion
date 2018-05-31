import torch

from torch import nn
import torch.nn.functional as F

from ..layers.pooling import PoolingLayer
from ..layers.layers import (
                             LinearAggregationLayer,
                             CharEncoder
                            )

from ..utils.torch import to_var, pack_forward


class CharEmbeddingLayer(nn.Module):
    def __init__(self, embeddings, use_cuda=True):
        super(CharEmbeddingLayer, self).__init__()
        self.embeddings = embeddings
        self.use_cuda = use_cuda
        self.embedding_dim = embeddings.embedding_dim

    def forward(self, np_batch):
        """np_batch: (batch_size, seq_len, word_len)"""
        batch = to_var(torch.LongTensor(np_batch),
                       use_cuda=self.use_cuda,
                       requires_grad=False)

        batch_size, seq_len, word_len = batch.size()
        batch = batch.view(batch_size, seq_len * word_len)

        emb_batch = self.embeddings(batch)

        emb_batch = emb_batch.view(batch_size, seq_len, word_len,
                                   self.embedding_dim)
        return emb_batch


class WordEmbeddingLayer(nn.Module):
    def __init__(self, embeddings, use_cuda=True):
        super(WordEmbeddingLayer, self).__init__()
        self.embeddings = embeddings
        self.use_cuda = use_cuda
        self.embedding_dim = embeddings.embedding_dim

    def forward(self, np_batch, char_batch=None, word_lengths=None):
        """np_batch: (batch_size, seq_len)"""
        batch = to_var(torch.LongTensor(np_batch),
                       use_cuda=self.use_cuda,
                       requires_grad=False)

        emb_batch = self.embeddings(batch)
        return emb_batch


class WordCharEncodingLayer(nn.Module):
    AGGREGATION_METHODS = ['cat', 'scalar_gate', 'vector_gate']

    def __init__(self, word_embeddings, char_embeddings,
                 char_hidden_size=50, word_char_aggregation_method='cat',
                 train_char_embeddings=True, use_cuda=True):

        aggregation_method = word_char_aggregation_method
        if aggregation_method not in self.AGGREGATION_METHODS:
            raise RuntimeError(f'{aggregation_method} method not recogized. '
                               f'Try one of {self.AGGREGATION_METHODS})')

        super(WordCharEncodingLayer, self).__init__()
        self.word_embeddings = word_embeddings
        self.char_embeddings = char_embeddings
        self.char_hidden_size = char_hidden_size
        self.aggregation_method = aggregation_method
        self.train_char_embeddings = train_char_embeddings
        self.use_cuda = use_cuda

        self.word_embedding_layer = WordEmbeddingLayer(word_embeddings,
                                                       use_cuda=self.use_cuda)
        self.char_embedding_layer = CharEmbeddingLayer(char_embeddings,
                                                       use_cuda=self.use_cuda)

        self.char_encoding_layer = CharEncoder(char_embeddings.embedding_dim,
                                               self.char_hidden_size,
                                               fw_bw_aggregation_method='linear_sum',
                                               bidirectional=True,
                                               train_char_embeddings=True,
                                               use_cuda=self.use_cuda)

        if self.aggregation_method == 'cat':
            # we add these dimensions because we are going to concatenate the vector reprs
            self.embedding_dim = self.char_encoding_layer.out_dim + word_embeddings.embedding_dim

        elif self.aggregation_method == 'scalar_gate':
            self.embedding_dim = self.char_encoding_layer.out_dim
            self.scalar_gate = nn.Linear(self.char_encoding_layer.out_dim, 1)

        elif self.aggregation_method == 'vector_gate':
            self.embedding_dim = self.char_encoding_layer.out_dim
            self.vector_gate = nn.Linear(self.char_encoding_layer.out_dim,
                                         self.char_encoding_layer.out_dim)

    def forward(self, word_batch, char_batch, word_lengths):
        emb_word_batch = self.word_embedding_layer(word_batch)
        emb_char_batch = self.char_embedding_layer(char_batch)

        char_lvl_word_repr = self.char_encoding_layer(emb_char_batch, word_lengths)

        if self.aggregation_method == 'cat':
            word_reprs = torch.cat([emb_word_batch, char_lvl_word_repr], 2)

        elif self.aggregation_method == 'scalar_gate':
            gate_result = F.sigmoid(self.scalar_gate(emb_word_batch))  # in [0; 1]
            word_reprs = (1.0 - gate_result) * emb_word_batch + gate_result * char_lvl_word_repr
            self.gate_result = gate_result

        elif self.aggregation_method == 'vector_gate':
            gate_result = F.sigmoid(self.vector_gate(emb_word_batch))  # in [0; 1]
            word_reprs = (1.0 - gate_result) * emb_word_batch + gate_result * char_lvl_word_repr
            self.gate_result = gate_result

        return word_reprs


class WordEncodingLayer(nn.Module):

    WORD_ENCODING_METHODS = ['embed', 'char_linear', 'char_lstm']

    @staticmethod
    def factory(word_encoding_method, *args, **kwargs):
        if word_encoding_method == 'embed':
            # FIXME: Hideous. Fix by using partials from functools or metaclasses
            kwargs.pop('char_embeddings')
            kwargs.pop('char_hidden_size')
            kwargs.pop('train_char_embeddings')
            kwargs.pop('word_char_aggregation_method')
            return WordEmbeddingLayer(*args, **kwargs)
        if word_encoding_method == 'char_linear':
            # return WordEmbeddingLayer(*args, **kwargs)
            raise NotImplementedError
        if word_encoding_method == 'char_lstm':
            return WordCharEncodingLayer(*args, **kwargs)

    def __init__(self, word_encoding_method, *args, **kwargs):
        super(WordEncodingLayer, self).__init__()
        self.word_encoding_method = word_encoding_method
        if self.word_encoding_method not in self.WORD_ENCODING_METHODS:
            raise AttributeError(f'`{self.word_encoding_method}` not '
                                 f'recognized. Try using '
                                 f'one of {self.WORD_ENCODING_METHODS}')

        self.word_encoding_layer = self.factory(self.word_encoding_method,
                                                *args,
                                                **kwargs)

        self.embedding_dim = self.word_encoding_layer.embedding_dim

    def __call__(self, *args, **kwargs):
        return self.word_encoding_layer(*args, **kwargs)


class StackedShortcutLSTM(nn.Module):
    """RNN for encoding context
       hidden_sizes: a list of len 3 containing lstm hidden sizes"""
    def __init__(self, embedding_dim, hidden_sizes=[512, 512, 512],
                 bidirectional=True, dropout=0.0, batch_first=True,
                 use_cuda=True):
        super(StackedShortcutLSTM, self).__init__()
        self.embedding_dim = embedding_dim
        self.hidden_sizes = hidden_sizes
        self.bidirectional = bidirectional
        self.num_dirs = 2 if bidirectional else 1
        self.dropout = dropout
        self.batch_first = batch_first
        self.use_cuda = use_cuda
        self.out_dim = self.hidden_sizes[-1] * self.num_dirs

        self.lstm_0 = nn.LSTM(input_size=self.embedding_dim,
                              hidden_size=self.hidden_sizes[0],
                              num_layers=1,
                              bidirectional=self.bidirectional,
                              dropout=0,
                              batch_first=self.batch_first)

        self.ext_hidden_size = (self.embedding_dim +
                                self.num_dirs * self.hidden_sizes[0])

        self.lstm_1 = nn.LSTM(input_size=self.ext_hidden_size,
                              hidden_size=self.hidden_sizes[1],
                              num_layers=1,
                              bidirectional=self.bidirectional,
                              dropout=0,
                              batch_first=self.batch_first)

        self.ext_hidden_size += self.num_dirs * self.hidden_sizes[1]

        self.lstm_2 = nn.LSTM(input_size=self.ext_hidden_size,
                              hidden_size=self.hidden_sizes[2],
                              num_layers=1,
                              bidirectional=self.bidirectional,
                              dropout=0,
                              batch_first=self.batch_first)

    def forward(self, emb_batch, lengths):
        # import ipdb; ipdb.set_trace()
        # dim(zero_state[0])= (num_dirs, batch, hidden_size)

        self.lstm_0.flatten_parameters()
        lstm_out_0 = pack_forward(self.lstm_0, emb_batch, lengths)

        new_batch_0 = torch.cat((emb_batch, lstm_out_0), dim=2)

        self.lstm_1.flatten_parameters()
        lstm_out_1 = pack_forward(self.lstm_1, new_batch_0, lengths)

        new_batch_1 = torch.cat((new_batch_0, lstm_out_1), dim=2)

        self.lstm_2.flatten_parameters()
        lstm_out_2 = pack_forward(self.lstm_2, new_batch_1, lengths)

        return lstm_out_2


class BLSTMEncoder(nn.Module):
    """
    Args:
        embedding_dim: """

    def __init__(self, embedding_dim, hidden_sizes=2048, num_layers=1,
                 bidirectional=True, dropout=0.0, batch_first=True, use_cuda=True):
        super(BLSTMEncoder, self).__init__()
        self.embedding_dim = embedding_dim
        self.hidden_size = hidden_sizes  # sizes in plural for compatibility
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.num_dirs = 2 if bidirectional else 1
        self.dropout = dropout
        self.batch_first = batch_first
        self.out_dim = self.hidden_size * self.num_dirs

        self.enc_lstm = nn.LSTM(self.embedding_dim, self.hidden_size,
                                num_layers=self.num_layers, bidirectional=True,
                                dropout=self.dropout)

    def is_cuda(self):
        # either all weights are on cpu or they are on gpu
        return 'cuda' in str(type(self.enc_lstm.bias_hh_l0.data))

    def forward(self, emb_batch, lengths):
        """Based on: https://github.com/facebookresearch/InferSent/blob/4b7f9ec7192fc0eed02bc890a56612efc1fb1147/models.py

           Take an embedded batch of dim (batch_size, seq_len, embedding_dim) and pass
           it through the RNN. Return a tensor of dim (batch_size, seq_len, out_dim)
           where out dim depends on the hidden dim of the RNN and its directions"""

        sent_output = pack_forward(self.enc_lstm, emb_batch, lengths)

        return sent_output


class SentenceEncodingLayer(nn.Module):

    SENTENCE_ENCODING_METHODS = ['stacked', 'lstm', 'infersent']

    @staticmethod
    def factory(sent_encoding_method, *args,  **kwargs):
        if sent_encoding_method == 'stacked':
            return StackedShortcutLSTM(*args, **kwargs)
        elif sent_encoding_method == 'lstm':
            raise NotImplementedError
            # return BaselineLSTM()
        elif sent_encoding_method == 'infersent':
            return BLSTMEncoder(*args, **kwargs)

    def __init__(self, sent_encoding_method, *args, **kwargs):
        super(SentenceEncodingLayer, self).__init__()
        self.sent_encoding_method = sent_encoding_method

        if self.sent_encoding_method not in self.SENTENCE_ENCODING_METHODS:
            raise AttributeError(f'`{self.sent_encoding_method}` not '
                                 f'recognized. Try using '
                                 f'one of {self.SENTENCE_ENCODING_METHODS}')

        self.sent_encoding_layer = self.factory(self.sent_encoding_method,
                                                *args,
                                                **kwargs)
        self.out_dim = self.sent_encoding_layer.out_dim

    def __call__(self, *args, **kwargs):
        return self.sent_encoding_layer(*args, **kwargs)


class NLIClassifier(nn.Module):
    """Args:
        embeddings: torch word embeddings
        """
    def __init__(self, num_classes, batch_size,
                 torch_embeddings=None,
                 char_embeddings=None,
                 word_encoding_method='embed',
                 word_char_aggregation_method=None,
                 sent_encoding_method='infersent',
                 hidden_sizes=None,
                 pooling_method='max',
                 batch_first=True,
                 dropout=0.0,
                 use_cuda=True):

        super(NLIClassifier, self).__init__()
        self.num_classes = num_classes
        self.batch_size = batch_size
        self.batch_first = batch_first
        self.dropout = dropout

        self.use_cuda = use_cuda

        self.pooling_method = pooling_method

        self.word_encoding_method = word_encoding_method
        self.sent_encoding_method = sent_encoding_method
        self.hidden_sizes = hidden_sizes

        self.char_embeddings = None
        if char_embeddings:
            self.char_embeddings = char_embeddings

        self.word_encoding_layer = WordEncodingLayer(self.word_encoding_method,
                                                     torch_embeddings,
                                                     char_embeddings=self.char_embeddings,
                                                     char_hidden_size=300,  # same dim as Glove Embeddings
                                                     word_char_aggregation_method=word_char_aggregation_method,
                                                     train_char_embeddings=True,
                                                     use_cuda=self.use_cuda)

        self.sent_encoding_layer = SentenceEncodingLayer(self.sent_encoding_method,
                                                         self.word_encoding_layer.embedding_dim,
                                                         hidden_sizes=self.hidden_sizes,
                                                         batch_first=self.batch_first,
                                                         use_cuda=self.use_cuda)

        self.pooling_layer = PoolingLayer(self.pooling_method)
        self.sent_aggregation_layer = LinearAggregationLayer()

        # we multiply by 4 because the LinearAggregationLayer concatenates
        # 4 vectors

        # self.dense_layer = nn.Linear(self.sent_encoding_layer.out_dim * 4,
        #                              self.num_classes)

        # self.dense_layer = nn.Sequential(nn.Linear(self.sent_encoding_layer.out_dim * 4,
        #                                            1600),
        #                                  nn.ReLU(), nn.Dropout(self.dropout),
        #                                  nn.Linear(1600, 1600),
        #                                  nn.ReLU(), nn.Dropout(self.dropout),
        #                                  nn.Linear(1600, self.num_classes))

        self.dense_layer = nn.Sequential(nn.Linear(self.sent_encoding_layer.out_dim * 4,
                                                   512),
                                         nn.ReLU(), nn.Dropout(self.dropout),
                                         nn.Linear(512, self.num_classes))

    def encode(self, batch, char_batch,
               sent_lengths, word_lengths,
               masks=None, embed_words=True):
        """ Encode a batch of ids into a sentence representation.

            This method exists for compatibility with facebook's senteval

            batch: padded batch of word indices if embed_words, else padded
                  batch of torch tensors corresponding to embedded word
                  vectors"""
        if embed_words:
            embedded = self.word_encoding_layer(batch, char_batch, word_lengths)
        else:
            embedded = batch
        sent_embedding = self.sent_encoding_layer(embedded, sent_lengths)
        agg_sent_embedding = self.pooling_layer(sent_embedding,
                                                lengths=sent_lengths,
                                                masks=masks)
        return agg_sent_embedding

    def forward(self, batch):
        # batch is created in Batch Iterator
        prems = batch['prems']
        hypos = batch['hypos']

        prem_sent_lengths = batch['prem_sent_lengths']
        prem_masks = batch['prem_masks']
        hypo_sent_lengths = batch['hypo_sent_lengths']
        hypo_masks = batch['hypo_masks']

        # TODO: to_var is going to happen for every batch every epoch which
        # makes this op O(num_batches * num_epochs). We could make it
        # O(num_batches) if we ran it once for every batch before training, but
        # this would limit our ability to shuffle the examples and re-create
        # the batches each epoch
        prem_sent_lengths = to_var(torch.FloatTensor(prem_sent_lengths),
                                   self.use_cuda,
                                   requires_grad=False)
        prem_masks = to_var(torch.FloatTensor(prem_masks),
                            self.use_cuda,
                            requires_grad=False)

        hypo_sent_lengths = to_var(torch.FloatTensor(hypo_sent_lengths),
                                   self.use_cuda,
                                   requires_grad=False)
        hypo_masks = to_var(torch.FloatTensor(hypo_masks),
                            self.use_cuda,
                            requires_grad=False)

        if self.char_embeddings:
            prem_char_sequences = batch['prem_char_sequences']
            hypo_char_sequences = batch['hypo_char_sequences']

            prem_word_lengths = batch['prem_word_lengths']
            hypo_word_lengths = batch['hypo_word_lengths']

            prem_char_masks = batch['prem_char_masks']
            hypo_char_masks = batch['hypo_char_masks']

            prem_word_lengths = to_var(torch.FloatTensor(prem_word_lengths),
                                       self.use_cuda,
                                       requires_grad=False)

            hypo_word_lengths = to_var(torch.FloatTensor(hypo_word_lengths),
                                       self.use_cuda,
                                       requires_grad=False)

            # prem_char_reprs = self.char_encoding_layer(prem_char_sequences)
            # prem_char_lvl_word_repr = self.char_level_word_encoder(prem_char_reprs,
            #                                                        prem_word_lengths)

        pooled_prem = self.encode(prems,
                                  char_batch=prem_char_sequences,
                                  sent_lengths=prem_sent_lengths,
                                  word_lengths=prem_word_lengths,
                                  masks=prem_masks)
        pooled_hypo = self.encode(hypos,
                                  char_batch=hypo_char_sequences,
                                  sent_lengths=hypo_sent_lengths,
                                  word_lengths=hypo_word_lengths,
                                  masks=hypo_masks)

        pair_repr = self.sent_aggregation_layer(pooled_prem, pooled_hypo)
        logits = self.dense_layer(pair_repr)

        ret_dict = {'logits': logits}

        return ret_dict
