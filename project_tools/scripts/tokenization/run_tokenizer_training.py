# Copyright (C) 2021 ServiceNow, Inc.
#
# run_tokenizer_training.py
"""Run tokenizer training

This is usually run via run_tokenizer_training.sh, which passes it the input file location.
The key output is a vocab.txt file that is saved in the hard-coded directory below (save_path),
    and named as wordpiece_geo_{filename_base}_{current_time} (via train_WordPiece).
"""

from datetime import datetime
from nrcan_p2.tokenization.custom_tokenizer import (train_WordPiece)
import argparse

# To pass arguments from command line
parser = argparse.ArgumentParser()
requiredNamed = parser.add_argument_group('Required named arguments')
requiredNamed.add_argument('--INPUT_FILES', nargs="+", help='list of input txt files')
args = parser.parse_args()

print(f"Training starting: {datetime.now(tz=None).strftime('%Y-%m-%d %H:%M:%S')}")

train_WordPiece(input_files = args.INPUT_FILES,
                save_path = '/nrcan_p2/data/06_models/tokenizers/geo_trained/')

print(f"Training done: {datetime.now(tz=None).strftime('%Y-%m-%d %H:%M:%S')}")
