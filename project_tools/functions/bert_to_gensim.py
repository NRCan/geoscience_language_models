# Copyright (C) 2021 ServiceNow, Inc.
"""Function used to embed a vocabulary of individual words into embeddings using BERT"""

import numpy as np
import pandas as pd
from gensim.models import KeyedVectors
import bert
import utils


def create_from_df(vocabulary, filename, model, tokenizer):

    """
    Creates dictionary of BERT embeddings for individual words in the vocabulary.
    """

    df = pd.DataFrame(vocabulary, columns=['vocab'])

    df['embedding'] = df.apply(lambda x: bert.bert_embedding(text=x['vocab'], tokenizer=tokenizer,
                                                             model=model, to_numpy=True), axis=1)

    vectors = df['embedding'].tolist()
    kv = KeyedVectors(vector_size=model.config.hidden_size)
    kv.add(entities=vocabulary, weights=vectors)
    kv.save(filename)