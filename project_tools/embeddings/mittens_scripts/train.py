# Copyright (C) 2021 ServiceNow, Inc.
""" Train a mittens model """

import csv
import numpy as np
import pickle
import argparse
from mittens import Mittens


def glove2dict(glove_filename):
    with open(glove_filename) as f:
        reader = csv.reader(f, delimiter=' ', quoting=csv.QUOTE_NONE)
        embed = {line[0]: np.array(list(map(float, line[1:])))
                for line in reader}
    return embed


def run_train(
    matrix_filename,
    vocab_filename,
    max_iter,
    original_embeddings_filename,
    vector_size,
    mittens_filename
):
    # Load cooccurrence matrix
    M = np.load(matrix_filename)

    # Load vocabulary
    infile = open(vocab_filename, 'rb')
    vocabulary = pickle.load(infile)
    infile.close()

    # Load pre-trained Glove embeddings
    original_embeddings = glove2dict(original_embeddings_filename)

    mittens_model = Mittens(n=config.VECTOR_SIZE, max_iter=max_iter)

    new_embeddings = mittens_model.fit(M, vocab=vocabulary, initial_embedding_dict=original_embeddings)

    np.save(mittens_filename, new_embeddings)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--MATRIX_FILENAME', help='input matrix file', required=True)
    parser.add_argument('--VOCAB_FILENAME', help='input vocab filename', required=True)
    parser.add_argument('--ORIGINAL_EMBEDDINGS_PATH', help='input glove filepath', required=True)
    parser.add_argument('--MAX_ITER', help='max iterations', required=True, type=int)
    parser.add_argument('--VECTOR_SIZE', help='vector size', required=True, type=int)
    parser.add_argument('--MITTENS_FILENAME', help='output filename', required=True)
    args = parser.parse_args()

    run_train(
        matrix_filename=args.MATRIX_FILENAME,
        vocab_filename=args.VOCAB_FILENAME,
        max_iter=args.MAX_ITER,
        original_embeddings_filename=args.ORIGINAL_EMBEDDINGS_PATH,
        vector_size=args.VECTOR_SIZE,
        mittens_filename=args.MITTENS_FILENAME
    )
