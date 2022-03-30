# Copyright (C) 2021 ServiceNow, Inc.
""" Utilities for evaluating nearest neighbors
"""

import numpy 
import pandas as pd

def nearest_neighbour_predict(model, word, k=10):
    """ Return the top k nearest neighboring words
        for the given word
    """
    result = model.most_similar(positive=[word], topn=k)
    return pd.DataFrame(result).rename(columns={0:'nn', 1:'similarity'})


def predict_all_nearest_neigbour(model, nn_df, k=10):
    """Predict all nearest neighbors of the words in the nn test set. 
        Note that these must be evaluated manually
    """
    
    nn_results = []
    for irow, row in nn_df.iterrows():
        
        try:
            result = nearest_neighbour_predict(model, word=row.Word.lower(), k=k)
            result['word'] = row.Word
            result['id'] = row.ID
        except KeyError as e:
            result = pd.DataFrame({'word': row.Word, 'id': row.ID})
            
        nn_results.append(result)
        
    return pd.concat(nn_results, axis=0)[['id', 'word', 'nn', 'similarity']]