# Copyright (C) 2021 ServiceNow, Inc.
"""Helper functions for the GloVe and BERT models and keyword expansion/search"""

import numpy as np
from gensim.models import KeyedVectors
import string
import re
import spacy
from spacy.lang.en.stop_words import STOP_WORDS

SPACY_MODEL_LG = spacy.load('en_core_web_lg')


def save_glove_model(text_path, save_path):

    """
    Saving a Glove model from a word2vec formatted .txt file.

    Creates two files; 1) save_path and 2) save_path.vectors.npy
    """

    model = KeyedVectors.load_word2vec_format(text_path)
    model.save(save_path)


def load_glove_model(file_path):

    """
    Load a Glove model from saved vectors. To load, the model needs two files:
    1) file_path
    2) file_path.vectors.npy

    param file_path: file path to the Glove model

    returns: Glove model stored as a Gensim KeyedVectors object
    """

    model = KeyedVectors.load(file_path)

    return model


def rm_punct(text: str):

    """
    Removes punctuation from text as part of text preprocessing pipeline.
    """

    punct = re.escape(string.punctuation)
    text = re.sub(f"[{punct}]", " ", text)
    text = re.sub(f" +", " ", text)
    return text


def lower(text: str):

    """
    Converts text to lowercase as part of text preprocessing pipeline.
    """

    return text.lower()


def tokenize_spacy_lg(text: str):

    """
    Performs tokenization of text as part of text preprocessing pipeline.
    """

    spacy_model = SPACY_MODEL_LG
    spacy_model.max_length = 1500000

    text_lines = text.split('\n')
    new_lines = []
    for line in text_lines:
        new_line = ' '.join([token.text for token in spacy_model(line, disable = ['ner', 'parser'])])
        new_lines.append(new_line)
    return '\n'.join(new_lines)


def rm_stopwords_spacy(text: str, stop_words=STOP_WORDS):

    """
    Removes stopwords from text as part of text preprocessing pipeline.
    """

    for word in stop_words:
        reg = re.compile(r'(?:(?<=\s)|(?<=^))' + word + r'(?:(?=\s)|(?=$))', re.IGNORECASE)
        text =  reg.sub(' ', text)
    text = re.sub(" +", " ", text)

    return text


def embed_text(text: str,
               model,
               agg_op: str = 'mean'):

    """
    Creates a fixed length embedding for variable length text using either the sum or the average of the embeddings
    from the individual words.

    @param text: title or paragraph
    @param model: model used for embedding
    @param agg_op: aggregation operation on individual word vectors, either 'mean' or 'sum'

    @return: returns vector embedding for input text
    """

    # If metadata is empty (for example, no title or abstract), return None
    if isinstance(text, str) and text.strip() != '':

        text = tokenize_spacy_lg(text)
        text = rm_stopwords_spacy(text)
        text = rm_punct(text)
        text = lower(text)

        text = text.split()

        vectors = []

        # Tracking words which were not in the embedding model vocabulary
        not_in_vocab = set()

        for word in text:
            try:
                vector = model[word]
                vectors.append(vector)
            except KeyError:
                not_in_vocab.add(word)

        # Return None if none of the processed text is in the embedding model vocabulary
        # Example of this case is a one-word title where the word is not in the embedding model
        if len(vectors) == 0:
            return None, list(not_in_vocab)

        embedding = np.sum(vectors, axis=0)

        if agg_op == 'mean':

            return embedding / len(vectors), list(not_in_vocab)

        elif agg_op == 'sum':

            return embedding, list(not_in_vocab)

    else:
        # Empty list returned for consistency with other return statements in function
        return None, list()


def find_similar_keyword_in_vocab(keyword,
                                  num_similar: int,
                                  model,
                                  similarity_score: bool = False):

    """
    Finds the top N most similar words to the keyword in the entire vocabulary.
    Calculated by default using cosine similarity.

    @param keyword: string or vector representation of keyword
    @param num_similar: number of similar words to return from the vocabulary
    @param model: embedding model

    @return: list of top N most similar words in order of descending similarity
    """

    # If input keyword is a string
    if isinstance(keyword, str):
        try:
            if similarity_score:
                return model.similar_by_word(word=keyword, topn=num_similar)
            else:
                return [word for word, similarity in model.similar_by_word(word=keyword, topn=num_similar)]
        except KeyError:
            return keyword

    # If input keyword is a vector, which can occur if input is an aggregated embedding from text such as an abstract
    elif isinstance(keyword, np.ndarray):
        if similarity_score:
            return model.similar_by_vector(vector=keyword, topn=num_similar)
        else:
            return [word for word, similarity in model.similar_by_vector(vector=keyword, topn=num_similar)]

