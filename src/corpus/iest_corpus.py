import os

from ..utils.io import load_or_create, read_jsonl, load_pickle
from .batch_iterator import BatchIterator
from .lang import Lang

from .. import config


class BaseCorpus(object):
    def __init__(self, paths_dict, mode='train', use_chars=True,
                 force_reload=False, train_data_proportion=1.0,
                 valid_data_proportion=1.0, batch_size=64,
                 shuffle_batches=False, batch_first=True):

        self.paths = paths_dict
        self.mode = mode

        self.use_chars = use_chars

        self.force_reload = force_reload

        self.train_data_proportion = train_data_proportion
        self.valid_data_proportion = valid_data_proportion

        self.batch_size = batch_size
        self.shuffle_batches = shuffle_batches
        self.batch_first = batch_first


class IESTCorpus(BaseCorpus):
    def __init__(self, *args, **kwargs):
        super(IESTCorpus, self).__init__(config.corpora_dict['iest'],
                                         *args, **kwargs)

        train_sents = open(self.paths['train']).readlines()

        # This assumes the data comes nicely separated by spaces. That's the
        # task of the tokenizer who should be called elsewhere
        self.train_sents = [s.rstrip().split(' ') for s in train_sents]

        dev_sents = open(self.paths['dev']).readlines()
        self.dev_sents = [s.rstrip().split(' ') for s in dev_sents]

        self.lang = Lang(self.train_sents)
