# coding=utf-8
SGNS_CHAR_PATH = 'data/word2vec/sgns.wiki.char'
SGNS_WORD_PATH = 'data/word2vec/sgns.wiki.word'

FIT_CHAR_PATH = 'data/fit_char_vectors.npz'
FIT_WORD_PATH = 'data/fit_word_vectors.npz'

LABEL_ID_PATH = 'data/level3_id.csv'
LABEL_PATH = 'data/level3_stat.txt'

TRAIN_PATH = 'data/train.tsv'
TEST_PATH = 'data/test.tsv'

TRAIN_CHAR_PATH = 'data/train_char.npz'
TRAIN_WORD_PATH = 'data/train_word.npz'

VOCAB_CHAR_PATH = 'data/vocab_char.txt'
VOCAB_WORD_PATH = 'data/vocab_word.txt'

STOP_WORDS_PATH = 'data/stop_words.txt'

LOG_DIR = 'log'
CHECKPOINTS_DIR = 'checkpoints'

# 字符级的文本最长长度
MAX_CHAR_TEXT_LENGTH = 48
# 词级的文本最长长度
MAX_WORD_TEXT_LENGTH = 27

VOCAB_COVERAGE = 0.9

VEC_DIM = 128
PRETRAINED_VEC_DIM = 300       # 预训练词向量的维度

NUM_CLASSES = 1258
