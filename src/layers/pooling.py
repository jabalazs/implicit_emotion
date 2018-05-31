import torch

from torch import nn


class PoolingLayer(nn.Module):

    POOLING_METHODS = ['mean', 'sum', 'last', 'max']

    @staticmethod
    def factory(pooling_method):
        if pooling_method == 'mean':
            return MeanPoolingLayer()
        elif pooling_method == 'sum':
            return SumPoolingLayer()
        elif pooling_method == 'last':
            return GatherLastLayer()
        elif pooling_method == 'max':
            return MaxPoolingLayer()

    def __init__(self, pooling_method):
        super(PoolingLayer, self).__init__()
        if pooling_method not in self.POOLING_METHODS:
            raise AttributeError(f'{pooling_method} not recognized. Try using '
                                 'one of {self.POOLING_METHODS}')

        self.pooling_layer = self.factory(pooling_method)

    def __call__(self, *args, **kwargs):
        return self.pooling_layer(*args, **kwargs)


class SumPoolingLayer(nn.Module):

    def __init__(self):
        super(SumPoolingLayer, self).__init__()

    def forward(self, sequences, **kwargs):

        """

        :param sequences:
            Size is  (batch_size, seq_len, hidden_dim)

        :param masks:
            Size is (batch_size, seq_len)

        :return:
            sum over seq_len after applying masks

        """

        batch_size = sequences.size(0)
        hidden_dim = sequences.size(2)
        # the masks are of dimension (batch_size, seq_len) and we need them
        # to be (batch_size, seq_len, hidden_x_dirs)
        masks = kwargs['masks']
        masks = masks.unsqueeze(2).repeat(1, 1, hidden_dim)
        masked_sequences = torch.mul(sequences, masks)

        # now, we want to sum the tensors along the sequence dimension which
        # will result in tensors of dimension (batch_size, hidden_x_dirs)
        aggregated_sequences = torch.sum(masked_sequences, 1)

        return aggregated_sequences


class MeanPoolingLayer(nn.Module):

    def __init__(self):
        super(MeanPoolingLayer, self).__init__()
        self.sum_pooling_layer = SumPoolingLayer()

    def forward(self, sequences, **kwargs):

        """

        :param sequences:
            Size is  (batch_size, seq_len, hidden_dim)

        :param lengths:
            Size is (batch_size)

        :param masks:
            Size is (batch_size, seq_len)

        :return:
            mean over seq_len after applying masks
        """

        batch_size = sequences.size(0)
        hidden_x_dirs = sequences.size(2)

        masks = kwargs['masks']
        sent_lengths = kwargs['lengths']

        aggregated_sequences = self.sum_pooling_layer(sequences, masks=masks)

        # for calculating the average we now broadcast the lengths:
        # (batch_size) -> (batch_size, hidden_x_dirs)
        sent_lengths = sent_lengths.unsqueeze(1)
        sent_lengths = sent_lengths.repeat(1, hidden_x_dirs)

        # finally we obtain the averages
        # (batch_size, hidden_x_dirs)
        sent_lengths = sent_lengths.float()
        aggregated_sequences = torch.div(aggregated_sequences, sent_lengths)

        return aggregated_sequences


class MaxPoolingLayer(nn.Module):

    def __init__(self):
        super(MaxPoolingLayer, self).__init__()

    def forward(self, sequences, **kwargs):

        """

        :param sequences:
            Size is  (batch_size, seq_len, hidden_dim)

        :param masks:
            Size is (batch_size, seq_len)

        :return:

        """

        batch_size = sequences.size(0)
        hidden_x_dirs = sequences.size(2)
        # the masks are of dimension (batch_size, seq_len) and we need them
        # to be (batch_size, seq_len, hidden_x_dirs)
        masks = kwargs['masks']
        masks = masks.unsqueeze(2).repeat(1, 1, hidden_x_dirs)
        masked_sequences = torch.mul(sequences, masks)

        # now, we want to obtain the maximum value along the sequence
        # dimension which will result in tensors of dimension
        # (batch_size, hidden_x_dirs)
        max_sequences, _ = torch.max(masked_sequences, 1)

        return max_sequences


class GatherLastLayer(nn.Module):

    def __init__(self, bidirectional=True):
        """

        Return the last hidden state of a tensor returned by an RNN

        """
        super(GatherLastLayer, self).__init__()
        self.bidirectional = bidirectional

    def forward(self, sequences, **kwargs):
        """

        Args:
            sequences: A Variable containing a 3D tensor of dimension
                (batch_size, seq_len, hidden_x_dirs)
            lengths: A Variable containing 1D LongTensor of dimension
                (batch_size)

        Return:
            A Variable containing a 2D tensor of the same type as sequences of
            dim (batch_size, hidden_x_dirs) corresponding to the concatenated
            last hidden states of the forward and backward parts of the input.
        """

        batch_size = sequences.size(0)
        seq_len = sequences.size(1)
        hidden_x_dirs = sequences.size(2)

        if self.bidirectional:
            assert hidden_x_dirs % 2 == 0
            single_dir_hidden = int(hidden_x_dirs / 2)
        else:
            single_dir_hidden = hidden_x_dirs

        lengths = kwargs['lengths']

        # Hacky way of checking the type of the input tensor
        if 'Float' in str(type(lengths.data)):
            lengths = lengths.long()

        lengths_fw = (lengths
                      .unsqueeze(1)
                      .unsqueeze(2)
                      .repeat(1, 1, single_dir_hidden))

        lengths_fw = lengths_fw - 1  # transform lengths to indices

        if not self.bidirectional:
            return torch.gather(sequences, 1, lengths_fw).squeeze()

        lengths_bw = (lengths_fw
                      .clone()
                      .zero_())

        # we want 2 chunks in the last dimension (2)
        out_fw, out_bw = torch.chunk(sequences, 2, 2)

        h_t_fw = torch.gather(out_fw, 1, lengths_fw)
        h_t_bw = torch.gather(out_bw, 1, lengths_bw)

        # -> (batch_size, hidden_x_dirs)
        last_hidden_out = torch.cat([h_t_fw, h_t_bw], 2).squeeze()
        return last_hidden_out
