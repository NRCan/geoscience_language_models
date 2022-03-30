# Copyright (C) 2021 ServiceNow, Inc.
""" Vectorization utilties """
import numpy as np


def convert_text_to_vector(text, model, method='sum'):
    """ Embed the tokens piece of text with a model.
        Tokens are produced by a simple whitespace split on the text
        if the text is provided as a string.
    
    :param text: text string or list
    :param model: word embedding model - must implement subscription by word
        e.g. mode['word']
    :param method: how to aggregate the individual token vectors
        sum - sum them
        mean - average them
        None - no aggregation, return a matrix of one vector per token
    """
    if type(text) == str:
        text = text.split()
    elif type(text) == list:
        pass
    else:
        raise ValueError('text must be a str or list')
        
    vectors = [model[word] for word in text if word in model]

    if len(vectors) == 0:
        vectors = np.zeros(shape=(model.vector_size,))
        return vectors
    try:
        vectors = np.stack(vectors)
    except Exception as e:
        print(e)
        print(vectors)

    if method == 'sum':
        vectors = np.sum(vectors, axis=0)
    elif method == 'mean':
        vectors = np.mean(vectors, axis=0)
    elif method == None:
        vectors = vectors
    else:
        raise ValueError(f'Unknown method: {method}')

    return vectors


def convert_dfcol_text_to_vector(df, col, model, method):
    """ Convert a text column of a df (col) to a vector, using 
        word embedding model model and vector aggregation method method.

    :param df: input dataframe
    :param col: text column to vectorize
    :param model: embedding model, must be subscriptable by word (e.g. model['word'])
    :param method: vector aggregation method

    :returns: an np.ndarray of shape (n_rows, n_vector_dim)
    """
    X = df[col].apply(lambda x: convert_text_to_vector(x, model, method=method))
    X = np.stack(X.values)
    return X