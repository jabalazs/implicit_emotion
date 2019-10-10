import sys

import torch
from .. import config
from ..utils.torch import to_var

sys.path.append(config.ALLENNLP_PATH)

from allennlp.modules.elmo import Elmo, batch_to_ids


class ElmoWordEncodingLayer(torch.nn.Module):
    def __init__(self, **kwargs):
        super(ElmoWordEncodingLayer, self).__init__()
        kwargs.pop('use_cuda')
        kwargs.pop('char_embeddings')
        kwargs.pop('char_hidden_size')
        kwargs.pop('train_char_embeddings')
        kwargs.pop('word_char_aggregation_method')
        self._embedder = Elmo(config.ELMO_OPTIONS, config.ELMO_WEIGHTS,
                              num_output_representations=1, **kwargs)
        self._embedder = self._embedder.cuda()
        # We know the output of ELMo with pre-trained weigths is of size 1024.
        # This would likely be different if we initialized `Elmo` with a custom
        # `module`, but we didn't test this, so for now it will be hardcoded.
        self.embedding_dim = 1024

    def forward(self, *args):
        """Sents a batch of N sentences represented as list of tokens"""

        # -1 is the raw_sequences element passed in the encode function of the
        # IESTClassifier
        # TODO: make this less hacky
        sents = args[-1]

        char_ids = batch_to_ids(sents)
        char_ids = to_var(char_ids,
                          use_cuda=True,
                          requires_grad=False)
        # returns a dict with keys: elmo_representations (list) and mask (torch.LongTensor)
        embedded = self._embedder(char_ids)

        embeddings = embedded['elmo_representations'][0]
        mask = embedded['mask']
        embeddings = to_var(embeddings,
                            use_cuda=True,
                            requires_grad=False)

        return embeddings, mask
