import os

DATA_PATH = 'data/'
PREPROCESSED_DATA_PATH = os.path.join(DATA_PATH, 'preprocessed')
RESULTS_PATH = os.path.join(DATA_PATH, 'results')
LOG_PATH = 'log/'

# EMBEDDINGS
EMBEDDINGS_DIR = os.path.join(DATA_PATH, 'word_embeddings')
SENNA_EMB_PATH = os.path.join(EMBEDDINGS_DIR, 'senna.txt')
GLOVE_EMB_PATH = os.path.join(EMBEDDINGS_DIR, 'glove.840B.300d.txt')
FASTTEXT_EMB_PATH = os.path.join(EMBEDDINGS_DIR, 'wiki.en.vec')

# CORPORA
TRAIN = os.path.join(PREPROCESSED_DATA_PATH, 'train_no_emojis.csv')
DEV = os.path.join(PREPROCESSED_DATA_PATH, 'dev_no_emojis.csv')

TRAIN_LABELS = os.path.join(PREPROCESSED_DATA_PATH, 'train_labels.csv')
DEV_LABELS = os.path.join(PREPROCESSED_DATA_PATH, 'dev_labels.csv')


# MAPPINGS
embedding_dict = {'senna': SENNA_EMB_PATH,
                  'glove': GLOVE_EMB_PATH,
                  'fasttext': FASTTEXT_EMB_PATH}

corpora_dict = {'iest': {'train': TRAIN,
                         'dev': DEV,
                         'test': None},
                }

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

LABEL_DICT = {'anger': 0, 'disgust': 1, 'fear': 2, 'joy': 3, 'sad': 4,
              'surprise': 5}

LABELS = ['anger', 'disgust', 'fear', 'joy', 'sad', 'surprise']

# DATABASE PARAMETERS
_DB_NAME = 'runs.db'

DATABASE_CONNECTION_STRING = 'sqlite:///' + os.path.join(RESULTS_PATH,
                                                         _DB_NAME)


JSON_KEYFILE_PATH = 'experiments database-a61b695b86f4.json'
# SERVER_NAME = open('server_name', 'r').read().strip()
