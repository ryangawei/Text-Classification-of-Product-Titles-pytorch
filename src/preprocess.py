# coding=utf-8
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
import pandas as pd
from nlputils.tokenizer import BasicTokenizer
from nlputils.vocab_generator import VocabGenerator
from config import *
import logging
logger = logging.getLogger(__name__)


def load_label2id(src_path):
    label2id = {}
    with open(src_path, 'r') as f:
        f.readline()
        while True:
            line = f.readline().strip()
            if line == '':
                break
            label, id = line.split(',')
            label2id[label] = int(id)
    return label2id


def preprocess_dataset(samples_path, vocab_path, npz_path, mode='char'):
    tokenizer = BasicTokenizer()
    tokenizer.load_stopwords(STOP_WORDS_PATH)
    gen = VocabGenerator(coverage=VOCAB_COVERAGE)

    label2id = load_label2id(LABEL_ID_PATH)

    if mode == 'char':
        vocab_path = VOCAB_CHAR_PATH
        max_length = MAX_CHAR_TEXT_LENGTH
    else:
        vocab_path = VOCAB_WORD_PATH
        max_length = MAX_WORD_TEXT_LENGTH

    pbar = tqdm(desc='Tokenizing all samples')
    seg_samples = []
    title_ids = []
    label_ids = []

    # Tokenize all samples.
    with open(samples_path, 'r', encoding='gb2312', errors='ignore') as f:
        header = f.readline()   # Omit the first line.
        while True:
            line = f.readline().strip()
            if line == '':
                break
            title, label = line.split('\t')
            label_ids.append(label2id[label])

            if mode == 'word':
                title_tokens = tokenizer.tokenize(title, no_stop_words=True)
            else:
                title_tokens = tokenizer.discard_stop_words(list(title))

            seg_samples.append(title_tokens)
            pbar.update()
    pbar.close()

    # Generate the vocab based on all samples.
    vocab = gen.generate_vocab(seg_samples)
    gen.save_vocab_to(vocab_path)
    logging.info('Save vocabulary to {}'.format(vocab_path))

    tokenizer.load_vocab(vocab)

    for sample in tqdm(seg_samples, desc='Converting samples into ids'):
        # Convert all sample tokens to ids.
        title_id = tokenizer.convert_tokens_to_ids(sample)
        title_ids.append(title_id)

    # Save all samples ids and their labels into .npz file.
    title_ids = np.asarray(title_ids)
    label_ids = np.asarray(label_ids)
    np.savez(npz_path, title_ids=title_ids, label_ids=label_ids)
    logging.info('Save dataset to {}'.format(npz_path))


def assign_label_id(label_path, id_path):
    """
    给标签分配id。

    :param label_path:
    :param id_path:
    :return:
    """
    readfile = open(label_path, 'r')
    ids = []
    id = 0
    for line in readfile.readlines():
        labelwords = line.split()[0:-1]     # 获取标签
        label = ' '.join(labelwords)
        ids.append((label, id))
        id += 1
    readfile.close()
    ids = pd.DataFrame(ids, columns=['label', 'id'])
    ids.to_csv(id_path, index=False, encoding='gb2312')


def show_length_distribution(npz_path):
    data = np.load(npz_path, allow_pickle=True)
    title_ids = data['title_ids']
    lengths = [len(x) for x in title_ids]

    plt.hist(lengths, bins=20, align='mid', edgecolor='black')
    plt.title('Title length distribution (Total %d)' % len(lengths))

    plt.xlabel('Title length')
    plt.ylabel('Number')
    plt.show()

    percentile = 0.85
    lengths = sorted(lengths)
    index = int(len(lengths) * percentile)
    print('The {} percentile is {}.'.format(percentile, lengths[index]))


def count_lines(fpath):
    with open(fpath, 'r', encoding='gb2312', errors='ignore') as f:
        i = 0
        while True:
            if f.readline() == '':
                break
            i += 1
        return i


def export_fit_embeddings(emb_path, npz_path, mode='char'):
    """
    Extract pre-trained word embeddings that appears in the vocabulary.
    Save as .npz.

    Words not appeared in the pre-trained vectors are randomly initialized.

    :param emb_path:
    :param mode:
    :return:
    """
    tokenizer = BasicTokenizer()
    if mode == 'char':
        tokenizer.load_vocab(VOCAB_CHAR_PATH)
    else:
        tokenizer.load_vocab(VOCAB_WORD_PATH)
    vocab = tokenizer.get_vocab()
    token2id = tokenizer.get_token2id()

    embeddings = np.random.randn(len(vocab), PRETRAINED_VEC_DIM)

    with open(emb_path, 'r', encoding='utf-8') as f:
        line_count, dim = f.readline().strip().split()
        logging.info('Total words {}, embedding dim {}.'.format(line_count, dim))  # First line is embedding info.
        for i in tqdm(range(int(line_count))):
            line = f.readline().strip().split()
            word = line[0]
            if word in vocab:
                vec = [float(x) for x in line[1:]]
                vec = np.asarray(vec, dtype=np.float32)  # Get the word vector.
                embeddings[token2id[word]] = vec

    np.savez(npz_path, embeddings=embeddings)
    logging.info('Fit embedding shape: {}'.format(embeddings.shape))
    logging.info('Export fit embeddings to {}'.format(npz_path))


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    preprocess_dataset(TRAIN_PATH, VOCAB_CHAR_PATH, TRAIN_CHAR_PATH, mode='char')
    preprocess_dataset(TRAIN_PATH, VOCAB_WORD_PATH, TRAIN_WORD_PATH, mode='word')
    # show_length_distribution(TRAIN_CHAR_PATH)
    # show_length_distribution(TRAIN_CHAR_PATH)
    export_fit_embeddings(SGNS_CHAR_PATH, FIT_CHAR_PATH, mode='char')
    export_fit_embeddings(SGNS_WORD_PATH, FIT_WORD_PATH, mode='word')
