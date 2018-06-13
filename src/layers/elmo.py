import sys

from torch import nn

from .. import config
from ..utils.torch import to_var

sys.path.append('/home/jabalazs/repos/allennlp/')

from allennlp.modules.elmo import Elmo, batch_to_ids


class ElmoWordEncodingLayer(object):
    def __init__(self, **kwargs):
        kwargs.pop('use_cuda')
        kwargs.pop('char_embeddings')
        kwargs.pop('char_hidden_size')
        kwargs.pop('train_char_embeddings')
        kwargs.pop('word_char_aggregation_method')
        self._embedder = Elmo(config.ELMO_OPTIONS, config.ELMO_WEIGHTS,
                              num_output_representations=1, **kwargs)
        self._embedder = self._embedder.cuda()
        self.embedding_dim = 1024

    def __call__(self, *args):
        """Sents a batch of N sentences represented as list of tokens"""

        # -1 is the raw_sequences element passed in the encode function of the
        # IESTClassifier
        sents = args[-1]
        char_ids = batch_to_ids(sents)
        char_ids = to_var(char_ids,
                          use_cuda=True,
                          requires_grad=False)
        # returns a dict with keys: elmo_representations (list) and mask (torch.LongTensor)
        embedded = self._embedder(char_ids)

        embeddings = embedded['elmo_representations'][0]
        embeddings = to_var(embeddings,
                            use_cuda=True,
                            requires_grad=False)

        return embeddings
