# Copyright (C) 2021 ServiceNow, Inc.
"""Keyword expansion and search using the BERT model"""

import os
import numpy as np
import pandas as pd
from collections import defaultdict
import torch
from gensim.models import KeyedVectors
from transformers import (
    AutoConfig,
    AutoTokenizer,
    AutoModel,
    DistilBertTokenizer)
from functions import utils


def load_bert(model_path):

    """
    model_path: path to the directory with the BERT model files

    Returns: the model, tokenizer and config objects

    """

    config = AutoConfig.from_pretrained(model_path)

    if os.path.isdir(model_path):
        tokenizer = DistilBertTokenizer.from_pretrained(model_path)
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_path)

    model = AutoModel.from_pretrained(model_path)
    model.eval()

    return model, tokenizer, config


def bert_tokenize(text, tokenizer, truncation=True, max_length=512):

    encoded_input = tokenizer(text, return_tensors='pt', truncation=truncation, max_length=max_length)

    return encoded_input


def bert_embedding(text, tokenizer, model, to_numpy=False):

    # Unlike Glove, BERT embedding function does not return the tokens not in the vocabulary because of the
    # wordpiece nature of the model

    # Return None for empty Geoscan metadata
    if isinstance(text, str) and text.strip() != '':

        encoded_input = bert_tokenize(text, tokenizer)

        with torch.no_grad():

            outputs = model(**encoded_input)

        # Extracting last hidden state
        token_embeddings = outputs.last_hidden_state
        # Removing batch dimension
        token_embeddings = torch.squeeze(token_embeddings, axis=0)
        # Remove hidden states of first token (CLS) and last token (SEP)
        token_embeddings = token_embeddings[1:-1]
        # Text embedding calculated as average of token embeddings
        text_embeddings = torch.mean(token_embeddings, dim=0)

        if to_numpy:
            return text_embeddings.numpy()
        else:
            return text_embeddings


def bert_similar_document_detection(df: pd.DataFrame,
                                    input_text: str,
                                    comparison_col: str,
                                    num_similar: int,
                                    tokenizer,
                                    model):

    """
    Takes as input text and returns a dataframe with the top N most similar articles, based on either similarity to the
    titles or to the abstracts of the articles.

    @param df: input pandas dataframe that has metadata embeddings stored
    @param input_text: string of text to embed and find similar documents to
    @param comparison_col: either embedded title or embedded abstract columns
    @param num_similar: number of similar documents to return
    @param tokenizer: BERT tokenizer object
    @param model: BERT model

    @return: returns a sorted dataframe with top N documents
    """

    assert isinstance(input_text, str), "Input text must be a string"

    embedding = bert_embedding(input_text, tokenizer, model, to_numpy=True)

    # Remove rows/articles with no embedding due to missing metadata
    df_result = df.copy()
    df_result.dropna(subset=[comparison_col], inplace=True)

    vector_list = np.stack(df_result[comparison_col].tolist(), axis=0)
    # Gensim model static method used for vector similarity calculation (768 vector size)
    df_result['similarity'] = KeyedVectors(vector_size=embedding.shape[0]).cosine_similarities(embedding, vector_list)

    return df_result.sort_values('similarity', ascending=False)[:num_similar]


def bert_text_to_keyword(text: str,
                         num_similar: int,
                         bert_model,
                         tokenizer,
                         bert_embedding_dict):

    """
    Finds the top N most similar words in the vocabulary to the input text.
    Calculated by default using cosine similarity.
    This differs from the Glove version in that the model used to embed and the model used to store embeddings are two
    different objects.

    @param text: keyword/title/paragraph as string
    @param num_similar: number of similar words to return
    @param bert_model: BERT embedding model
    @param tokenizer: BERT tokenizer object
    @param bert_embedding_dict: KeyedVectors object storing BERT-generated embeddings

    @return: List of top N most similar words in order of descending similarity
    """

    assert isinstance(text, str), "Input text must be a string"

    embedding = bert_embedding(text, tokenizer, bert_model, to_numpy=True)

    return utils.find_similar_keyword_in_vocab(keyword=embedding, num_similar=num_similar, model=bert_embedding_dict)


def bert_keyword_expansion(keywords: str,
                           num_similar: int,
                           bert_model,
                           tokenizer,
                           bert_embedding_dict):

    """
    Keyword expansion outputs top N most similar words from vocabulary.

    @param keywords: string of comma-separated words to find keywords for
    @param num_similar: number of similar words to return
    @param bert_model: BERT embedding model
    @param tokenizer: BERT tokenizer object
    @param bert_embedding_dict: KeyedVectors object storing BERT-generated embeddings

    @return: list of top N most similar words in order of descending similarity
    """

    assert isinstance(keywords, str)

    keywords = utils.rm_punct(keywords)
    keywords = utils.lower(keywords)

    # Assuming input is a string of space separated words
    keyword_list = set(keywords.split())

    # Dictionary used to store similarity scores for top keywords
    scores_dict = defaultdict(int)

    for keyword in keyword_list:

        # Check if keyword is in the BERT embedding dictionary
        # If not, we create a vector representation of it first
        if keyword not in bert_embedding_dict.vocab:
            keyword = bert_embedding(keyword, tokenizer, bert_model, to_numpy=True)

        # Returns a list of tuples in the form (word, similarity score)
        result = utils.find_similar_keyword_in_vocab(keyword=keyword, num_similar=num_similar,
                                                     model=bert_embedding_dict, similarity_score=True)
        for word, score in result:
            # Skipping similar words that already in the list of keywords provided by user
            if word in keyword_list:
                continue
            else:
                # Keeping the maximum similarity score for each word
                scores_dict[word] = max(scores_dict[word], score)

    sorted_results = sorted(scores_dict.items(), key=lambda kv: kv[1], reverse=True)[:num_similar]

    return [word for word, score in sorted_results]
