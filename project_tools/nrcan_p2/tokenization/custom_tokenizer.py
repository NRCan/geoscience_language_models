# Copyright (C) 2021 ServiceNow, Inc.
#
# custom_tokenizer.py
"""Functions to train a WordPiece tokenizer from custom data

This is mainly used by run_tokenizer_training.py (and in notebooks for development and experimentation)
"""

import os.path
from datetime import datetime
import sys

from tokenizers import Tokenizer
from tokenizers.models import WordPiece
from tokenizers import normalizers
from tokenizers.normalizers import Lowercase, NFD, StripAccents
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.processors import TemplateProcessing
from tokenizers.trainers import WordPieceTrainer

def train_WordPiece(input_files, save_path):
    """
        Trains a WordPiece tokenizer with BERT pre-/post-processing and saves:
            (1) vocab list (tokens)
            (2) model properly initialized with unknown token [UNK]

        :param input_files: List of strings with length >=1 indicating path to input file(s) for training
        :param save_path: String indicating path to folder for saving trained tokenizer model
    """
    # If input_files was given as a string, convert to a list for downstream compatibility
    if isinstance(input_files, str):
        print('Argument `input_files` given as string; converting to list.')
        input_files = [input_files]

    # Check that input file(s) exist(s); exit if not
    for filename in input_files:
        if os.path.isfile(filename):
            print('Input file exists: ' + filename)
        else:
            print(f'Input file does not exist: {filename}')
            sys.exit(f'Input file not found:  {filename}')

    # Check that output folder exists; exit if not
    if os.path.isdir(save_path):
        print(f'Output directory exists: {save_path}')
    else:
        print(f'Output directory does not exist:  {save_path}')
        sys.exit(f'Output directory not found:  {save_path}')

    # Determine filename for saving model:
    # 'wordpiece_geo' + name of first input data file + datetimestamp
    # If filename is 'train', get name from enclosing folder instead.
    filename_base = os.path.splitext(input_files[0].rsplit(sep='/', maxsplit=1)[-1])[0]
    if filename_base == 'train':
        filename_base = os.path.splitext(input_files[0].rsplit(sep='/', maxsplit=2)[-2])[0]

    current_time = datetime.now(tz=None).strftime('%Y%m%d_%H%M%S')

    save_filename = f'wordpiece_geo_{filename_base}_{current_time}'
    print(f'Model will be saved with filename {save_filename}')

    # Instantiate tokenizer and create pipeline for pre- and post-processing
    tokenizer = Tokenizer(WordPiece())
    tokenizer.normalizer = normalizers.Sequence([NFD(), Lowercase(), StripAccents()])
    tokenizer.pre_tokenizer = Whitespace()
    tokenizer.post_processor = TemplateProcessing(
        single='[CLS] $A [SEP]',
        pair='[CLS] $A [SEP] $B:1 [SEP]:1',
        special_tokens=[
            ('[CLS]', 1),
            ('[SEP]', 2),
        ],
    )

    # Train
    trainer = WordPieceTrainer(
        vocab_size=30522,
        special_tokens=['[UNK]', '[CLS]', '[SEP]', '[PAD]', '[MASK]']
    )
    text_files = input_files
    tokenizer.train(trainer, text_files)

    # Save model + vocab
    model_files = tokenizer.model.save(
        save_path,
        save_filename
    )

    # Reload with unknown token
    tokenizer.model = WordPiece.from_file(*model_files, unk_token='[UNK]')

    # Save full model
    tokenizer.save(os.path.join(save_path, save_filename + '.json'))
