import copy

import torch.nn as nn

from .transformer import (Encoder,
                          EncoderLayer,
                          MultiHeadedAttention,
                          PositionwiseFeedforward,
                          PositionalEncoding)


class TransformerEncoder(nn.Module):

    """Transformer Encoder"""

    def __init__(self, embedding_dim, hidden_sizes, num_layers=6, num_heads=8,
                 dropout=0.1, batch_first=True, use_cuda=True):
        """Take a batch of representations and add context transformer-style

        Parameters
        ----------
        embedding_dim : TODO
        hidden_sizes : TODO
        num_layers : TODO, optional
        num_heads : TODO, optional
        dropout : TODO, optional
        batch_first: TODO, optional
        use_cuda : TODO, optional


        """
        if not batch_first:
            raise NotImplementedError

        super(TransformerEncoder, self).__init__()
        self.embedding_dim = embedding_dim
        self.hidden_sizes = hidden_sizes
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dropout = dropout
        self.use_cuda = use_cuda

        self.out_dim = embedding_dim

        #  FIXME: I don't know how will deepcopies work within a pytorch module
        # <2018-06-25 12:06:59, Jorge Balazs>
        c = copy.deepcopy
        attn = MultiHeadedAttention(self.num_heads, self.embedding_dim)
        ff = PositionwiseFeedforward(self.embedding_dim, self.hidden_sizes,
                                     self.dropout)
        position = PositionalEncoding(self.embedding_dim, self.dropout)
        self.encoder = Encoder(
            EncoderLayer(embedding_dim, c(attn), c(ff), dropout), self.num_layers
        )
        self.positional_embedding = c(position)

        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, emb_batch, masks=None, lengths=None):
        """Add context to a batch of vectors

        Parameters
        ----------
        emb_batch : torch.FloatTensor, dim(batch_size, seq_len, hidden_dim)
        mask : torch.Floattensor, dim(batch_size, seq_len)
        lengths : kept for compatibility with other layers

        Returns
        -------
        A torch.FloatTensor of dim(batch_size, seq_len, hidden_dim) containing
        context-enriched vectors

        """
        # for compatibility with Annotated Transformer implementation
        masks = masks.unsqueeze(1)

        return self.encoder(self.positional_embedding(emb_batch), masks)
