# Copyright (C) 2021 ServiceNow, Inc.
""" Produce the coocurrence matrix for mittens """

import itertools
from collections import Counter
import numpy as np
import pickle
import argparse


def read_corpus(input_text):
    with open(input_text, 'r') as files:
        return [f.split() for f in files]


def create_vocab(corpus, vocab_min_count, max_vocab_size):
    words = list(itertools.chain.from_iterable(corpus))

    counter = Counter(words)

    if max_vocab_size is not None:
        counter = dict(itertools.takewhile(lambda i: i[1] >= vocab_min_count, counter.most_common(max_vocab_size)))
    else:
        counter = dict(itertools.takewhile(lambda i: i[1] >= vocab_min_count, counter.most_common()))
    return counter


def create_matrix(corpus, vocab, window_size):
    num_words = len(vocab)

    M = np.zeros((num_words, num_words))
    vocabulary = {w: i for i, w in enumerate(vocab)}

    for k, doc in enumerate(corpus, 1):

        for pos, token in enumerate(doc):

            try:
                i = vocabulary[token]
                start = max(0, pos - window_size)
                end = min(len(doc), pos + window_size + 1)

                for pos2 in range(start, end):

                    if pos2 == pos:
                        continue

                    try:

                        j = vocabulary[doc[pos2]]
                        d = abs(pos - pos2)

                        M[i, j] += 1 / d

                    except KeyError:

                        continue

            except KeyError:

                continue

    return vocabulary, M


def run_cooccur(
    input_filename,
    vocab_min_count,
    max_vocab_size,
    window_size,
    matrix_filename,
    vocab_filename
):
    # Creating list of documents from corpus
    corpus = read_corpus(input_filename)

    # Creating dictionary of words and their occurrence count in the corpus
    words_dic = create_vocab(corpus, vocab_min_count=vocab_min_count, max_vocab_size=max_vocab_size)

    vocab = list(words_dic.keys())

    vocabulary, M = create_matrix(corpus=corpus, vocab=vocab, window_size=window_size)

    np.save(matrix_filename, M)

    f = open(vocab_filename, 'wb')
    pickle.dump(vocabulary, f)
    f.close()    
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--INPUT_TEXT_FILENAME', help='input file', required=True)
    parser.add_argument('--VOCAB_MIN_COUNT', help='min token count to be include in the vocab', required=True, type=int)
    parser.add_argument('--MAX_VOCAB_SIZE', help='max vocab size', required=True, type=int)
    parser.add_argument('--WINDOW_SIZE', help='window size', required=True, type=int)
    parser.add_argument('--MATRIX_FILENAME', help='output matrix file', required=True)
    parser.add_argument('--VOCAB_FILENAME', help='output vocab filename', required=True)
    args = parser.parse_args()

    run_cooccur(
        input_filename=args.INPUT_TEXT_FILENAME,
        vocab_min_count=args.VOCAB_MIN_COUNT,
        max_vocab_size=args.MAX_VOCAB_SIZE,
        window_size=args.WINDOW_SIZE,
        matrix_filename=args.MATRIX_FILENAME,
        vocab_filename=args.VOCAB_FILENAME,
    )


