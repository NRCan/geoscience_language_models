# Copyright (C) 2021 ServiceNow, Inc.
""" Run mittens training """

import pathlib
import json
import argparse 
from embeddings.mittens_scripts.cooccur import run_cooccur
from embeddings.mittens_scripts.train import run_train

parser = argparse.ArgumentParser()
parser.add_argument('--OUTPUT_FOLDER', help='output folder', required=True)

parser.add_argument('--INPUT_TEXT_FILENAME', help='input file', required=True)
parser.add_argument('--VOCAB_MIN_COUNT', help='min token count to be include in the vocab', required=True, type=int)
parser.add_argument('--MAX_VOCAB_SIZE', help='max vocab size', default=None, type=int)
parser.add_argument('--WINDOW_SIZE', help='window size', required=True, type=int)

parser.add_argument('--ORIGINAL_EMBEDDINGS_PATH', help='input glove filepath', required=True)
parser.add_argument('--MAX_ITER', help='max iterations', required=True, type=int)
parser.add_argument('--VECTOR_SIZE', help='vector size', required=True, type=int)

args = parser.parse_args()

# get the output folder
OUTPUT_FOLDER = pathlib.Path(args.OUTPUT_FOLDER)
print(f'Output folder: {OUTPUT_FOLDER}')
if not OUTPUT_FOLDER.exists():
    OUTPUT_FOLDER.mkdir(parents=False, exist_ok=False)

# write the config file
output_config_file = str(OUTPUT_FOLDER / 'config.conf')
print(f'Writing config file... {output_config_file}')
print(args)
with open(output_config_file, 'w') as f:
    json.dump(vars(args), f, indent=4)

MATRIX_FILENAME = str(OUTPUT_FOLDER / 'matrix.npy')
VOCAB_FILENAME = str(OUTPUT_FOLDER / 'vocab.pkl')
MITTENS_FILENAME = str(OUTPUT_FOLDER / 'new_embeddings.npy')

run_cooccur(
    input_filename=args.INPUT_TEXT_FILENAME,
    vocab_min_count=args.VOCAB_MIN_COUNT,
    max_vocab_size=args.MAX_VOCAB_SIZE,
    window_size=args.WINDOW_SIZE,
    matrix_filename=MATRIX_FILENAME,
    vocab_filename=VOCAB_FILENAME,
)

run_train(
    matrix_filename=MATRIX_FILENAME,
    vocab_filename=VOCAB_FILENAME,
    max_iter=args.MAX_ITER,
    original_embeddings_filename=args.ORIGINAL_EMBEDDINGS_PATH,
    vector_size=args.VECTOR_SIZE,
    mittens_filename=MITTENS_FILENAME
)
