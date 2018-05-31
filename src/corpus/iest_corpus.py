import os

from ..utils.io import load_or_create, read_jsonl, load_pickle
from .batch_iterator import BatchIterator
from .lang import Lang

from .. import config


class BaseCorpus(object):
    def __init__(self, paths_dict, mode='train', use_chars=True,
                 force_reload=False, train_data_proportion=1.0,
                 dev_data_proportion=1.0, batch_size=64,
                 shuffle_batches=False, batch_first=True):

        self.paths = paths_dict
        self.mode = mode

        self.use_chars = use_chars

        self.force_reload = force_reload

        self.train_data_proportion = train_data_proportion
        self.dev_data_proportion = dev_data_proportion

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

        self.train_id_sents = self.lang.sents2ids(self.train_sents)
        self.dev_id_sents = self.lang.sents2ids(self.dev_sents)

        self.train_char_id_sents = self.lang.sents2char_ids(self.train_sents)
        self.dev_char_id_sents = self.lang.sents2char_ids(self.dev_sents)

        train_labels = open(config.TRAIN_LABELS).readlines()
        self.train_labels = [l.rstrip() for l in train_labels]
        self.train_id_labels = [config.LABEL_DICT[label]
                                for label in self.train_labels]

        dev_labels = open(config.DEV_LABELS).readlines()
        self.dev_labels = [l.rstrip() for l in dev_labels]
        self.dev_id_labels = [config.LABEL_DICT[label]
                              for label in self.dev_labels]

        self.train_ids = range(len(self.train_id_sents))
        self.dev_ids = range(len(self.dev_id_sents))

        train_examples = zip(self.train_ids,
                             self.train_id_sents,
                             self.train_char_id_sents,
                             self.train_id_labels)

        dev_examples = zip(self.dev_ids,
                           self.dev_id_sents,
                           self.dev_char_id_sents,
                           self.dev_id_labels)

        self.train_examples = [{'id': ex[0],
                                'sequence': ex[1],
                                'char_sequence': ex[2],
                                'label': ex[3]} for ex in train_examples]

        self.dev_examples = [{'id': ex[0],
                              'sequence': ex[1],
                              'char_sequence': ex[2],
                              'label': ex[3]} for ex in dev_examples]

        self.train_batches = BatchIterator(self.train_examples,
                                           self.batch_size,
                                           data_proportion=self.train_data_proportion,
                                           shuffle=True,
                                           batch_first=self.batch_first,
                                           use_chars=self.use_chars)

        self.dev_batches = BatchIterator(self.dev_examples,
                                         self.batch_size,
                                         data_proportion=self.dev_data_proportion,
                                         shuffle=False,
                                         batch_first=self.batch_first,
                                         use_chars=self.use_chars)
