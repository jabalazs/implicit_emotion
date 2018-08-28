import os

HOME_DIR = os.environ['HOME']
ALLENNLP_PATH = 'allennlp/'

DATA_PATH = 'data/'
CACHE_PATH = '.cache'
PREPROCESSED_DATA_PATH = os.path.join(DATA_PATH, 'preprocessed')
RESULTS_PATH = os.path.join(DATA_PATH, 'results')
LOG_PATH = 'log/'

# EMBEDDINGS
EMBEDDINGS_DIR = os.path.join(DATA_PATH, 'word_embeddings')
FASTTEXT_EMB_PATH = os.path.join(EMBEDDINGS_DIR, 'wiki.en.vec')
GLOVE_EMB_PATH = os.path.join(EMBEDDINGS_DIR, 'glove.840B.300d.txt')

ELMO_OPTIONS = os.path.join(EMBEDDINGS_DIR, 'elmo_2x4096_512_2048cnn_2xhighway_options.json')
ELMO_WEIGHTS = os.path.join(EMBEDDINGS_DIR, 'elmo_2x4096_512_2048cnn_2xhighway_weights.hdf5')

GLOVE_TWITTER_200_EMB_PATH = os.path.join(EMBEDDINGS_DIR, 'glove.twitter.27B.200d.txt')
GLOVE_TWITTER_100_EMB_PATH = os.path.join(EMBEDDINGS_DIR, 'glove.twitter.27B.100d.txt')
GLOVE_TWITTER_50_EMB_PATH = os.path.join(EMBEDDINGS_DIR, 'glove.twitter.27B.50d.txt')
GLOVE_TWITTER_25_EMB_PATH = os.path.join(EMBEDDINGS_DIR, 'glove.twitter.27B.25d.txt')

# CORPORA
TRAIN = os.path.join(PREPROCESSED_DATA_PATH, 'train_no_emojis.csv')
DEV = os.path.join(PREPROCESSED_DATA_PATH, 'dev_no_emojis.csv')

TRAIN_EMOJI = os.path.join(PREPROCESSED_DATA_PATH, 'train.csv')
DEV_EMOJI = os.path.join(PREPROCESSED_DATA_PATH, 'dev.csv')

TRAIN_LABELS = os.path.join(PREPROCESSED_DATA_PATH, 'train_labels.csv')
DEV_LABELS = os.path.join(PREPROCESSED_DATA_PATH, 'dev_labels.csv')

TEST = os.path.join(PREPROCESSED_DATA_PATH, 'test_no_emojis.csv')
TEST_EMOJI = os.path.join(PREPROCESSED_DATA_PATH, 'test.csv')
TEST_LABELS = os.path.join(PREPROCESSED_DATA_PATH, 'test_labels.csv')

TRAIN_POS = os.path.join(PREPROCESSED_DATA_PATH, 'train.csv.tagged.pos')
DEV_POS = os.path.join(PREPROCESSED_DATA_PATH, 'dev.csv.tagged.pos')
TEST_POS = os.path.join(PREPROCESSED_DATA_PATH, 'test.csv.tagged.pos')


# MAPPINGS
embedding_dict = {
     'fasttext': FASTTEXT_EMB_PATH,
     'glove': GLOVE_EMB_PATH,
     'glove_twitter_200': GLOVE_TWITTER_200_EMB_PATH,
     'glove_twitter_100': GLOVE_TWITTER_100_EMB_PATH,
     'glove_twitter_50': GLOVE_TWITTER_50_EMB_PATH,
     'glove_twitter_25': GLOVE_TWITTER_25_EMB_PATH,
}

corpora_dict = {
    'iest': {
        'train': TRAIN,
        'dev': DEV,
        'test': TEST
    },
    'iest_emoji': {
        'train': TRAIN_EMOJI,
        'dev': DEV_EMOJI,
        'test': TEST_EMOJI
    },
}

pos_corpora_dict = {
    'iest_emoji': {
        'train': TRAIN_POS,
        'dev': DEV_POS,
        'test': TEST_POS
    }
}

LABEL_PATHS = {'train': TRAIN_LABELS,
               'dev': DEV_LABELS,
               'test': TEST_LABELS}

WRITE_MODES = {'none': None,
               'file': 'FILE',
               'db': 'DATABASE',
               'both': 'BOTH'}

PAD_ID = 0
UNK_ID = 1
NUM_ID = 2
URL_ID = 3
# SOS_ID =
# EOS_ID =

# Specific to IEST dataset
USR_ID = 4
TRIGGERWORD_ID = 5


PAD_TOKEN = '__PAD__'
UNK_TOKEN = '__UNK__'
NUM_TOKEN = '__NUM__'
URL_TOKEN = '__URL__'
# SOS_TOKEN = '__SOS__'
# EOS_TOKEN = '__EOS__'

# These should be just like the ones appearing in the input dataset (these are
# different to the originals because of preprocessing)
USR_TOKEN = '__USERNAME__'
TRIGGERWORD_TOKEN = '__TRIGGERWORD__'

SPECIAL_TOKENS = {
                  PAD_TOKEN: PAD_ID,
                  UNK_TOKEN: UNK_ID,
                  NUM_TOKEN: NUM_ID,
                  URL_TOKEN: URL_ID,
                  USR_TOKEN: USR_ID,
                  TRIGGERWORD_TOKEN: TRIGGERWORD_ID,
                  }

UNK_CHAR_ID = 0
UNK_CHAR_TOKEN = 'あ'

SPECIAL_CHARS = {
                 UNK_CHAR_TOKEN: UNK_CHAR_ID
                }

LABEL2ID = {'anger': 0, 'disgust': 1, 'fear': 2, 'joy': 3, 'sad': 4,
            'surprise': 5}

ID2LABEL = {value: key for key, value in LABEL2ID.items()}

LABELS = ['anger', 'disgust', 'fear', 'joy', 'sad', 'surprise']

# DATABASE PARAMETERS
_DB_NAME = 'runs.db'

DATABASE_CONNECTION_STRING = 'sqlite:///' + os.path.join(RESULTS_PATH,
                                                         _DB_NAME)


JSON_KEYFILE_PATH = 'experiments-database-8ee4da525610.json'

try:
    SERVER_NAME = open('server_name', 'r').read().strip()
except FileNotFoundError:
    pass

SPREADSHEET_NAME = 'iest_experiments'
