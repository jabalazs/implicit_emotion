import os
import json

import substring_nli.config as config
from substring_nli.corpus.tree import Tree
from substring_nli.corpus.lang import Lang
from substring_nli.corpus.batch_iterator import BatchIterator
from substring_nli.utils.io import load_or_create


class NLICorpusSentence(object):

    def __init__(self, string, tree_string, binary_tree_string, id=None):
        self.string = string
        self.tree_string = tree_string
        self.binary_tree_string = binary_tree_string
        tree = Tree.fromstring(tree_string)
        self.tokens, self.pos_tags = zip(*tree.pos())
        if id:
            self.id = id

    def __contains__(self, item):
        return item in self.tokens

    def __iter__(self):
        return iter(self.tokens)

    def __getitem__(self, item):
        return self.tokens[item]

    def __len__(self):
        return len(self.tokens)

    def count(self, value):
        return self.tokens.count(value)


class NLICorpusExample(object):

    def __init__(self, example):
        self.annotator_labels = example['annotator_labels']
        self.caption_id = example.get('captionID', None)
        self.pair_id = example.get('pairID', None)
        self.genre = example.get('genre', None)
        self.gold_label = example['gold_label']
        self.sentence_1 = NLICorpusSentence(example['sentence1'],
                                            example['sentence1_parse'],
                                            example['sentence1_binary_parse'])

        self.sentence_2 = NLICorpusSentence(example['sentence2'],
                                            example['sentence2_parse'],
                                            example['sentence2_binary_parse'])


class NLICorpus(object):
    """
    Class to read SNLI corpus.

    Note: The class construction is not optimized for efficiency but for easier
    understanding. We iterate several times over the data instead of doing
    everything in one pass.
    """

    # path = None
    label_dict = config.SNLI_LABEL_DICT

    def __str__(self):
        return self.__class__.__name__

    def __repr__(self):
        return str(self)

    @property
    def name(self):
        return str(self)

    @property
    def num_classes(self):
        return len(self.label_dict)

    def __init__(self, paths, mode='train', chars=False):
        """
        paths: dict containing train, dev, test jsonl filepaths
        """

        self.mode = mode
        self.chars = chars
        self.paths = paths

        self._path_root, self._ext = os.path.splitext(self.paths[mode])

        self.pos_id_tuples = []

        raw_examples_pickle_path = self._path_root + '.pickle'
        self.raw_examples = load_or_create(raw_examples_pickle_path, self._read)

        tuples_pickle_path = self._path_root + '_tuples.pickle'
        self.tuples = load_or_create(tuples_pickle_path, self._create_tuples)

        # NOTE: here the lang object created will always be tied to the training
        # set, regardless of the mode with which the instance of this class was
        # initialized
        path_root, ext = os.path.splitext(self.paths['train'])
        lang_pickle_path = path_root + '_lang.pickle'
        self.lang = load_or_create(lang_pickle_path, self._create_lang)

        # Used when creating embeddings
        self.lang.set_word_frequency_threshold(low_freq_threshold=0)

        id_tuples_pickle_path = self._path_root + '_idtuples.pickle'
        self.id_tuples = load_or_create(id_tuples_pickle_path, self._create_id_tuples)

        if self.chars:
            char_id_tuples_pickle_path = self._path_root + '_char_idtuples.pickle'
            self.char_id_tuples = load_or_create(char_id_tuples_pickle_path,
                                                 self._create_char_id_tuples)

        # if self.lang.has_pos_tags:
        #     pos_id_tuples_pickle_path = self._path_root + '_pos_idtuples.pickle'
        #     self.pos_id_tuple = load_or_create(pos_id_tuples_pickle_path,
        #                                        self._create_pos_id_tuples)

    def _read(self):
        raw_examples = []
        with open(self.paths[self.mode], "r") as f:
            for line in f.readlines():
                example = json.loads(line)
                try:
                    if example['gold_label'] == '-':
                        print('Ignoring example with "-" label: {}'
                              ''.format(example["pairID"]))
                        continue
                    # Dirty hack for reading test set
                    elif example['gold_label'] == 'hidden':
                        example['gold_label'] = 'neutral'
                except KeyError:
                    pass
                raw_examples.append(NLICorpusExample(example))
        return raw_examples

    def _create_tuples(self):
        tuples = []

        for idx, example in enumerate(self.raw_examples):
            premise = example.sentence_1
            hypothesis = example.sentence_2
            label_idx = self.label_dict[example.gold_label]
            try:
                sent_id = example.pair_id
            except AttributeError:
                sent_id = idx
            tuples.append((premise, hypothesis, label_idx, sent_id))

            return tuples

    def _create_lang(self):
        """Create the language belonging to the corpus

        For now we're going to hardcode the train corpus language
        as the universal one"""

        lang = Lang(self)
        for hypothesis, premise, label_idx, sent_id in self.tuples:
            lang.read_add_sentence(premise)
            lang.read_add_sentence(hypothesis)
        return lang

    def _create_id_tuples(self):

        id_tuples = []
        for tupl in self.tuples:
            premise = tupl[0]
            hypothesis = tupl[1]
            label_id = tupl[2]
            sent_id = tupl[3]
            premise_ids = self.lang.sentence2index(premise)
            hypothesis_ids = self.lang.sentence2index(hypothesis)
            id_tuples.append((premise_ids, hypothesis_ids, label_id, sent_id))
        return id_tuples

    def _create_char_id_tuples(self):

        char_id_tuples = []
        for tupl in self.id_tuples:
            premise_ids = tupl[0]
            hypothesis_ids = tupl[1]
            label_id = tupl[2]
            sent_id = tupl[3]

            premise_char_ids = self.lang.idsentence2charids(premise_ids)
            hypothesis_char_ids = self.lang.idsentence2charids(hypothesis_ids)

            char_id_tuples.append((premise_char_ids, hypothesis_char_ids,
                                   label_id, sent_id))

        return char_id_tuples

    def get_torch_embeddings(self, embeddings_path):
        return self.lang.get_torch_embeddings(embeddings_path)

    def get_batch_iterator(self, batch_size, data_proportion=1.0,
                           shuffle=False, max_prem_len=None, batch_first=False,
                           use_char_embeddings=False, allowed_labels=()):

        char_id_tuples = None
        allowed_label_ids = [config.SNLI_LABEL_DICT[label]
                             for label in allowed_labels]
        if use_char_embeddings:
            char_id_tuples = self.char_id_tuples

        return BatchIterator(self.id_tuples,
                             batch_size,
                             data_proportion=data_proportion,
                             shuffle=shuffle,
                             max_prem_len=max_prem_len,
                             batch_first=batch_first,
                             char_id_tuples=char_id_tuples,
                             allowed_label_ids=allowed_label_ids)
