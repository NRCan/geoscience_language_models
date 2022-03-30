# Copyright (C) 2021 ServiceNow, Inc.
"""Keyword expansion and search using the GloVe model"""

import numpy as np
import pandas as pd
from collections import defaultdict
from functions import utils


def keyword_expansion(keywords: str,
                      num_similar: int,
                      model):
    """
    Keyword expansion outputs top N most similar words from vocabulary.

    @param keywords: string of comma-separated words to find keywords for
    @param num_similar: number of similar words to return
    @param model: embedding model

    @return: list of top N most similar words in order of descending similarity and a list of words not in vocabulary
    """

    assert isinstance(keywords, str)

    keywords = utils.rm_punct(keywords)
    keywords = utils.lower(keywords)

    # Assuming input is a string of space separated words
    keyword_list = set(keywords.split())

    # Dictionary used to store similarity scores for top keywords
    scores_dict = defaultdict(int)
    not_in_vocab = set()

    for keyword in keyword_list:

        # Returns a list of tuples in the form (word, similarity score)
        result = utils.find_similar_keyword_in_vocab(keyword=keyword, num_similar=num_similar,
                                                     model=model, similarity_score=True)
        if isinstance(result, list):

            for word, score in result:
                # Skipping similar words that already in the list of keywords provided by user
                if word in keyword_list:
                    continue
                else:
                    # Keeping the maximum similarity score for each word
                    scores_dict[word] = max(scores_dict[word], score)
        else:

            not_in_vocab.add(result)

    sorted_results = sorted(scores_dict.items(), key=lambda kv: kv[1], reverse=True)[:num_similar]

    return [word for word, score in sorted_results], list(not_in_vocab)


def text_to_keywords(text: str,
                     num_similar: int,
                     model,
                     agg_op: str = 'mean'):

    """
    Finds the top N most similar words in the vocabulary to the input text.
    Calculated by default using cosine similarity.

    @param text: keyword/title/paragraph as string
    @param num_similar: number of similar words to return
    @param model: embedding model
    @param agg_op: aggregation operation on individual word vectors, either 'mean' or 'sum'

    @return: List of top N most similar words in order of descending similarity and a list of words not in vocabulary
    """

    assert isinstance(text, str), "Input text must be a string"

    embedding, not_in_vocab = utils.embed_text(text=text, model=model, agg_op=agg_op)

    return utils.find_similar_keyword_in_vocab(keyword=embedding, num_similar=num_similar, model=model), not_in_vocab


def similar_document_detection(df: pd.DataFrame,
                               input_text: str,
                               comparison_col: str,
                               num_similar: int,
                               model,
                               agg_op: str = 'mean'):

    """
    Takes as input text and returns a dataframe with the top N most similar articles, based on either similarity to the
    titles or to the abstracts of the articles.

    @param df: input pandas dataframe that has metadata embeddings stored
    @param input_text: string of text to embed and find similar documents to
    @param comparison_col: either embedded title or embedded abstract columns
    @param num_similar: number of similar documents to return
    @param model: embedding model
    @param agg_op: aggregation operation on individual word vectors, either 'mean' or 'sum'

    @return: returns a sorted dataframe with top N documents
    """

    assert isinstance(input_text, str), "Input text must be a string"

    embedding, not_in_vocab = utils.embed_text(text=input_text, model=model, agg_op=agg_op)

    # Remove rows/articles with no embedding due to missing metadata
    df_result = df.copy()
    df_result.dropna(subset=[comparison_col], inplace=True)

    vector_list = np.stack(df_result[comparison_col].tolist(), axis=0)

    df_result['similarity'] = model.cosine_similarities(embedding, vector_list)

    return df_result.sort_values('similarity', ascending=False)[:num_similar]

