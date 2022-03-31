# Copyright (C) 2021 ServiceNow, Inc.
#
# create_geo_tokenizers.py
"""Create geology tokenizers by modifying vocab of pretrained tokenizer

See BERT_Tokenizer_Training.ipynb for development and experimentation.
"""
from transformers import BertTokenizer
import os
import sys

# Read in generic BERT vocab.txt
bert_vocab_file = os.path.join('/nrcan_p2/data/06_models/tokenizers/distilbert-base-uncased',
                          'vocab.txt')
with open(bert_vocab_file) as f:
    temp = f.read()
    bert = temp.splitlines()

# Read in vocab from geology tokenizer
vocab_file = os.path.join('/nrcan_p2/data/06_models/tokenizers/geo_trained',
                          'wordpiece_geo_EAIDown.xml_processed_nosentences_20210223_161710-vocab.txt')
with open(vocab_file) as f:
    temp = f.read()
    geo = temp.splitlines()

assert len(bert) == len(geo), f'Pretrained and geology tokenizers are not the same length (pretrained: {len(bert)}; geo: {len(geo)})'
print(f'Both the pretrained and geology tokenizers have {len(bert)} tokens in their vocabularies.')

# Determine which tokens are new
geo_new = list(set(geo).difference(set(bert)))
print(f'There are {len(geo_new)} new tokens.')

# Order tokens in list by order in vocab.txt
geo_new_ordered = [token for token in geo if token in geo_new]

# Determine indices of [unusedXX] tokens in generic tokenizers
unused_indices = [bert.index(token) for token in bert if '[unused' in token]
assert len(unused_indices) == 994, f'Pretrained tokenizer vocab has {len(unused_indices)} unused tokens instead of 994.'

def subst_geo_for_unused(count):
    """
        Function to substitute new tokens from the geo tokenizer
            for unused tokens in the pretrained bert tokenizer.
        Returns the new vocab list.

        :param count: Number of unused tokens to substitute with geo tokens (<=994)
    """
    assert count<=994, 'count must not be more than 994'
    vocab = bert.copy()
    for i in range(0,count):
        vocab[unused_indices[i]] = geo_new_ordered[i]
    return vocab

def write_new_vocab(count, token_dir):
    """
        Function to write a new vocab list (vocab.txt) to the appropriate directory for the
            new tokenizer (based on the number of substituted tokens).

        :param count: Number of unused tokens to substitute with geo tokens (<=994)
        :param token_dir: Directory for writing tokenizer; expected to be /nrcan_p2/data/06_models/tokenizers/bert_geo/bert_geo_{count}/
    """
    bert_geo_count_vocab = subst_geo_for_unused(count)
    tokenizer_file = os.path.join(token_dir, 'vocab.txt')
    with open(tokenizer_file, 'w') as f:
        for i, token in enumerate(bert_geo_count_vocab):
            if i < (len(bert_geo_count_vocab)-1):
                f.write(f'{token}\n')
            else:
                f.write(f'{token}')
        print(f'New vocab list written to {tokenizer_file}')

# Copy other appropriate files
def make_tokenizer(count):
    """
        Creates new tokenizer using trained geo tokenizer and pretrained BERT tokenizer:
            - Checks for tokenizer directory ('/nrcan_p2/data/06_models/tokenizers/bert_geo/bert_geo_{count}/')
                - If it exists, does nothing.
                - If it doesn't exist:
                    - Makes a new directory
                    - Copies in the appropriate tokenizer files
                    - Creates and writes a new vocab file

        :param count: Number of unused tokens to substitute with geo tokens (<=994)
    """

    token_dir = f'/nrcan_p2/data/06_models/tokenizers/bert_geo/bert_geo_{count}/'

    if os.path.exists(token_dir):
        sys.exit(f'Tokenizer directory already exists. Will not recreate tokenizer.')
    else:
        os.makedirs(token_dir)
        print(f'{count} tokens will be added to the tokenizer vocabulary.')

        # Copy tokenizer files
        cmd_special_tokens = f'cp /nrcan_p2/data/06_models/tokenizers/distilbert-base-uncased/special_tokens_map.json {os.path.join(token_dir, "special_tokens_map.json")}'
        cmd_tok_config = f'cp /nrcan_p2/data/06_models/tokenizers/distilbert-base-uncased/tokenizer_config.json {os.path.join(token_dir, "tokenizer_config.json")}'
        os.system(cmd_special_tokens)
        os.system(cmd_tok_config)

        # Make and write vocab
        write_new_vocab(count, token_dir)

make_tokenizer(300)
