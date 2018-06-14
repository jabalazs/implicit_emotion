import numpy as np

import torch

from torch import nn
import torch.nn.functional as F

from ..layers.pooling import PoolingLayer

from ..layers.layers import (
    CharEmbeddingLayer,
    WordEmbeddingLayer,
    CharEncoder,
    InfersentAggregationLayer,
)

from ..layers.elmo import ElmoWordEncodingLayer

from ..utils.torch import to_var, pack_forward, to_torch_embedding


class AttentionLayer(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(AttentionLayer, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim

        v = np.random.uniform(-0.01, 0.01, size=(self.output_dim))
        v = torch.from_numpy(v).float().requires_grad_()
        self.v = torch.nn.Parameter(v)

        self.linear_layer = torch.nn.Linear(self.input_dim, self.output_dim)

    def forward(self, sequence, vector, char_masks):
        """sequence: a batch of sequences of vector representations of dim
                     (B, S, word_len, D1), batch, sequence length, word length,
                     hidden dim

           vector: a batch of vectors these are the ones we want to compare
           the sequences against. dim: (B, S, D2),

           char_masks: (B, S, word_len)"""

        B, S, word_len, D1 = sequence.size()
        _, _, D2 = vector.size()
        # (output_dim) -> (B, S, word_len, output_dim)
        v = self.v.expand(B, S, word_len, -1)

        # (B, S, D2) -> (B, S, word_len, D2)
        vector = vector.unsqueeze(2).expand(-1, -1, word_len, -1)

        # -> (B, S, word_len, D1 + D2)
        concatted = torch.cat((sequence, vector), dim=3)

        # (B, S, word_len, D1 + D2) -> (B, S, word_len, output_dim)
        linearized = self.linear_layer(concatted)

        linearized = torch.tanh(linearized)

        linearized = linearized.view(B * S * word_len, self.output_dim).unsqueeze(2)
        v = v.view(B * S * word_len, self.output_dim).unsqueeze(1)
        score = torch.bmm(v, linearized).unsqueeze(1).unsqueeze(1)
        score = score.view(B, S, word_len)

        # We transform the elements corresponding to paddings to -inf
        inf_batch_mask = (1 - char_masks).byte()
        score.masked_fill_(inf_batch_mask, -1e16)

        # (B * S, 1, word_len)
        alphas = F.softmax(score, dim=2).unsqueeze(2).view(B * S, 1, word_len)

        # (B * S, 1, D1)
        weighted = torch.bmm(alphas, sequence.view(B * S, word_len, D1))

        weighted = weighted.squeeze(1).view(B, S, D1)

        return weighted


class WordCharEncodingLayer(nn.Module):
    AGGREGATION_METHODS = ['cat', 'scalar_gate', 'vector_gate', 'infersent', 'attention']

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

        fw_bw_aggregation_method = 'linear_sum'
        if self.aggregation_method == 'attention':
            # if we use attention we need the char encoding layer to return the
            # non-aggregated character-level representations.
            fw_bw_aggregation_method = None

        self.char_encoding_layer = CharEncoder(char_embeddings.embedding_dim,
                                               self.char_hidden_size,
                                               fw_bw_aggregation_method=fw_bw_aggregation_method,
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

        elif self.aggregation_method == 'infersent':
            self.embedding_dim = 2 * (self.char_encoding_layer.out_dim + word_embeddings.embedding_dim)
            self.infersent_aggregation = InfersentAggregationLayer()

        elif self.aggregation_method == 'attention':
            self.embedding_dim = self.char_encoding_layer.out_dim
            self.attention_layer = AttentionLayer(self.char_encoding_layer.out_dim + word_embeddings.embedding_dim,
                                                  self.char_encoding_layer.out_dim)

    def forward(self, word_batch, char_batch, word_lengths, char_masks, *args):
        """args is here for compatibility with other encoding layers, specifically
        for accepting the raw_sequences parameter used in the elmo encoding layer"""
        # FIXME: really need to find a way to avoid having dummy parameters such as
        # args here
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

        elif self.aggregation_method == 'infersent':
            word_reprs = self.infersent_aggregation(emb_word_batch, char_lvl_word_repr)

        elif self.aggregation_method == 'attention':
            word_reprs = self.attention_layer(char_lvl_word_repr,
                                              emb_word_batch,
                                              char_masks)

        return word_reprs


class WordEncodingLayer(nn.Module):

    WORD_ENCODING_METHODS = ['embed', 'char_linear', 'char_lstm', 'elmo']

    @staticmethod
    def factory(word_encoding_method, *args, **kwargs):
        if word_encoding_method in ['embed', 'elmo']:
            if word_encoding_method == 'embed':
                return WordEmbeddingLayer(*args, **kwargs)
            elif word_encoding_method == 'elmo':
                return ElmoWordEncodingLayer(**kwargs)

        if word_encoding_method == 'char_linear':
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

    def __repr__(self):
        s = '{name}('
        s += 'method={word_encoding_method}'
        s += ')'
        return s.format(name=self.__class__.__name__,
                        **self.__dict__)


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
                 bidirectional=True, dropout=0.0, batch_first=True,
                 use_cuda=True):
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
        # FIXME: Avoid calls to data()
        # either all weights are on cpu or they are on gpu
        return 'cuda' in str(type(self.enc_lstm.bias_hh_l0.data))

    def forward(self, emb_batch, lengths):
        sent_output = pack_forward(self.enc_lstm, emb_batch, lengths)
        return sent_output


class SentenceEncodingLayer(nn.Module):

    SENTENCE_ENCODING_METHODS = ['stacked', 'lstm', 'bilstm']

    @staticmethod
    def factory(sent_encoding_method, *args,  **kwargs):
        if sent_encoding_method == 'stacked':
            return StackedShortcutLSTM(*args, **kwargs)
        elif sent_encoding_method == 'lstm':
            raise NotImplementedError
            # return BaselineLSTM()
        elif sent_encoding_method == 'bilstm':
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


class IESTClassifier(nn.Module):
    """ Classifier for the Implicit Emotion Shared Task

    Parameters
    ----------
    num_classes: int
    batch_size : int
    embedding_matrix: numpy.ndarray
    char_embedding_matrix: numpy.ndarray
    word_encoding_method: str
    word_char_aggregation_method: str
    sent_encoding_method: str
    hidden_sizes:
    sent_enc_layers: int
    pooling_method: str
    batch_first: bool
    dropout: float
    lstm_dropout: float
    use_cuda: bool
        """
    def __init__(self, num_classes, batch_size,
                 embedding_matrix=None,
                 char_embedding_matrix=None,
                 word_encoding_method='embed',
                 word_char_aggregation_method=None,
                 sent_encoding_method='bilstm',
                 hidden_sizes=None,
                 sent_enc_layers=1,
                 pooling_method='max',
                 batch_first=True,
                 dropout=0.0,
                 lstm_dropout=0.0,
                 use_cuda=True):

        super(IESTClassifier, self).__init__()
        self.num_classes = num_classes
        self.batch_size = batch_size
        self.batch_first = batch_first
        self.dropout = dropout
        self.lstm_dropout = lstm_dropout

        self.use_cuda = use_cuda

        self.pooling_method = pooling_method

        self.word_encoding_method = word_encoding_method
        self.sent_encoding_method = sent_encoding_method
        self.hidden_sizes = hidden_sizes
        self.sent_enc_layers = sent_enc_layers

        self.char_embeddings = None
        if char_embedding_matrix is not None:
            self.char_embeddings = to_torch_embedding(char_embedding_matrix)

        torch_embeddings = to_torch_embedding(embedding_matrix)

        self.word_encoding_layer = WordEncodingLayer(
            self.word_encoding_method,
            torch_embeddings,
            char_embeddings=self.char_embeddings,
            char_hidden_size=torch_embeddings.embedding_dim,  # same dim as the word embeddings
            word_char_aggregation_method=word_char_aggregation_method,
            train_char_embeddings=True,
            use_cuda=self.use_cuda
        )

        self.sent_encoding_layer = SentenceEncodingLayer(
            self.sent_encoding_method,
            self.word_encoding_layer.embedding_dim,
            hidden_sizes=self.hidden_sizes,
            num_layers=self.sent_enc_layers,
            batch_first=self.batch_first,
            use_cuda=self.use_cuda,
            dropout=self.lstm_dropout
        )

        self.pooling_layer = PoolingLayer(self.pooling_method)

        self.dense_layer = nn.Sequential(
            nn.Linear(self.sent_encoding_layer.out_dim, 512),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(512, self.num_classes)
        )

    def encode(self, batch, char_batch,
               sent_lengths, word_lengths,
               masks=None, char_masks=None, embed_words=True,
               raw_sequences=None):
        """ Encode a batch of ids into a sentence representation.

            This method exists for compatibility with facebook's senteval

            batch: padded batch of word indices if embed_words, else padded
                   batch of torch tensors corresponding to embedded word
                   vectors

            embed_words: whether to pass the input through an embedding layer
                         or not
        """

        if embed_words:
            embedded = self.word_encoding_layer(batch, char_batch, word_lengths,
                                                char_masks, raw_sequences)
        else:
            embedded = batch
        sent_embedding = self.sent_encoding_layer(embedded, sent_lengths)
        agg_sent_embedding = self.pooling_layer(sent_embedding,
                                                lengths=sent_lengths,
                                                masks=masks)
        return agg_sent_embedding

    def forward(self, batch):
        # batch is created in Batch Iterator
        sequences = batch['sequences']
        raw_sequences = batch['raw_sequences']
        sent_lengths = batch['sent_lengths']
        masks = batch['masks']

        # TODO: to_var is going to happen for every batch every epoch which
        # makes this op O(num_batches * num_epochs). We could make it
        # O(num_batches) if we ran it once for every batch before training, but
        # this would limit our ability to shuffle the examples and re-create
        # the batches each epoch
        sent_lengths = to_var(torch.FloatTensor(sent_lengths),
                              self.use_cuda,
                              requires_grad=False)
        masks = to_var(torch.FloatTensor(masks),
                       self.use_cuda,
                       requires_grad=False)

        if self.char_embeddings:
            char_sequences = batch['char_sequences']
            word_lengths = batch['word_lengths']
            char_masks = batch['char_masks']

            word_lengths = to_var(torch.LongTensor(word_lengths),
                                  self.use_cuda,
                                  requires_grad=False)

            char_masks = to_var(torch.LongTensor(char_masks),
                                self.use_cuda,
                                requires_grad=False)

        sent_vec = self.encode(sequences,
                               char_batch=char_sequences,
                               sent_lengths=sent_lengths,
                               word_lengths=word_lengths,
                               masks=masks,
                               char_masks=char_masks,
                               raw_sequences=raw_sequences)

        logits = self.dense_layer(sent_vec)

        ret_dict = {'logits': logits}

        return ret_dict
