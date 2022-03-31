# Tokenization / Geology-specific tokenizer

`README` for `nrcan_p2/scripts/tokenization`

This README covers the workflow for creating a geology-specific tokenizer. This workflow feeds into the DistilBERT model training workflow.  I.e., once geology-specific tokenizers exist, they can be used for training DistilBERT models.

---

## Scripts

### Overview

At a high level, the workflow is the following:
* Train a tokenizer with a very clean, small, geology-specific dataset
* Select x tokens from this tokenizer to add to the pretrained tokenizer (where, in various experiments, `x = [250, 500, 994]`)

### Script details
Data prerequisite:
* `'/nrcan_p2/data/03_primary/metadata/EAIDown.xml_processed_nosentences.txt'` should already exist.

If one were starting from scratch, the scripts would be run as follows:

1.
    -  `run_tokenizer_training.py`
    - Trains a tokenizer using the specified input file
    - Saves to the `/nrcan_p2/data/06_models/tokenizers/geo_trained/` directory
    - Saves two files as `wordpiece_geo_{filename_base}_{current_time}` + `-vocab.txt` or `.json`; both contain the tokenizer vocabulary
    - Uses `train_WordPiece` imported from `nrcan_p2.tokenization.custom_tokenizer`
2. `download_save_pretrained_tokenizer.py.py`
  - This downloads and saves the pretrained distibert-base-uncased to `/nrcan_p2/data/06_models/tokenizers/distilbert-base-uncased/`. This only ever needs to be run once, just so that the files exist to create the modified/geology tokenizers.
3. `create_geo_tokenizers.py`
  - This script creates geology tokenizers by extending the vocabulary of the pretrained distilbert-base-uncased tokenizer. It does this by replacing the first x of the 994 `[unusedXX]` tokens in the pretrained tokenizer (so x is at most 994).
  - The script is hard-coded to create 3 tokenizers, i.e. with 250, 500, and 994 replaced tokens. One could absolutely modify it to create tokenizers with different numbers of replaced tokens.
  - The tokenizers are then saved to `/nrcan_p2/data/06_models/tokenizers/bert_geo/bert_geo_{count}/` where `count` is the number of replaced tokens.

Once these tokenizers exist, they can be read into model training code using `BertTokenizer.from_pretrained()` (after `from transformers import BertTokenizer`).

**NOTE:** Before training, ensure that the provided model and tokenizer paths are accessible to the filesystem on which the code is running. This may require changing the paths in the code. In particular, note input and output paths for custom tokenizer training (1. above), and input paths for using a custom tokenizer to extend the pretrained tokenizer (2. and 3. above).

---

## Notebooks

Two notebooks were used for developing and experimenting with this code, and for analyzing intermittent results (e.g., figuring out the best approach for adding new tokens to a tokenizer, reading in a custom tokenizer, etc.). Both are in the directory `nrcan_p2/notebooks/tokenization`.

* `BERT_DataPrep_Tokenizer_Mockup.ipynb`
  - This includes code and examples for training a custom tokenizer from scratch. This was initial testing, and is the less useful of the two notebooks.
* `BERT_Tokenizer_Training.ipynb`
  - This includes development, experimentation, examples, etc. for basically everything described in this README.

---
