# Copyright (C) 2021 ServiceNow, Inc.
#
# download_save_pretrained_tokenizer.py
"""Download and save the pretrained DistilBERT tokenizer

This only needs to be run once so that the files exist.
"""
from transformers import BertTokenizer

bert_tokenizer = BertTokenizer.from_pretrained('distilbert-base-uncased', do_lower_case=True)
bert_tokenizer.save_pretrained('/nrcan_p2/data/06_models/tokenizers/distilbert-base-uncased/')
