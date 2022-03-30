# Copyright (C) 2021 ServiceNow, Inc.
""" Utilities for evaluating analogies

"""
import numpy as np
import pandas as pd

def analogy_predict(model, quartet, i=2, topn=10):
    """Return the result of an analogy test, given a quartet
    
    :param quartet: 4 item list of strings in (man, king, woman, queen) type ordering
    :param i: the index of the quartet item to predict
    :param topn: the number of most similar words to return
    
    :returns: a df of the topn most similar words to the result of the vector manipulation that 
        should result in item index i of the quartet, 
        with columns ['word', 'similarity']
    
    :example: 
    analogy_predict(('woman', 'queen', 'man', king'), i=0, topn=2)
    >   word | similarity
    > 0 woman| 0.894704
    > 1 girl | 0.848953
    """
    if i == 0:
        result = model.most_similar(positive=[quartet[1], quartet[2]], negative=[quartet[3]], topn=topn)
    elif i == 1:
        result = model.most_similar(positive=[quartet[0], quartet[3]], negative=[quartet[2]], topn=topn)
    elif i == 2:
        result = model.most_similar(positive=[quartet[0], quartet[3]], negative=[quartet[1]], topn=topn)
    elif i == 3:
        result = model.most_similar(positive=[quartet[2], quartet[1]], negative=[quartet[0]], topn=topn)
    else:
        raise ValueError(f'Invalid i {i}. Must be from 0-3.')
        
    return pd.DataFrame(result).rename(columns={0: 'word', 1: 'similarity'})


def analogy_distance(model, quartet, i=2):
    """Return the distance ("error") between the result of an analogy test (the result of the 
       vector subtraction), and the correct word in vector space, given a quartet
       
    :param quartet: 4 item list of strings in (man, king, woman, queen) type ordering
    :param i: the index of the quartet item to predict
    
    :returns: a df with 1 row, containing the word of the quartet being estimated, and 
        the euclidean distance in vector space
        with columns ['word', 'distance']
    """
    if i == 0:
        result = model[quartet[1]] + model[quartet[2]] - model[quartet[3]]
    elif i == 1:
        result = model[quartet[0]] + model[quartet[3]] - model[quartet[2]]
    elif i == 2:
        result = model[quartet[0]] + model[quartet[3]] - model[quartet[1]]
    elif i == 3:
        result = model[quartet[2]] + model[quartet[1]] - model[quartet[0]]
    else:
        raise ValueError(f'Invalid i {i}. Must be from 0-3.')

    distance = model.distances(result, [quartet[i]])
    return pd.DataFrame({'word': quartet[i], 'distance': distance}, index=[0])


def evaluate_all_analogies(model, analogies_df, formations=[0,1,2,3], topns=[1, 5, 10, 100]):
    """Evaluate all analogies from an analogies test set, calculating the specified permutations
        and accuracy in the topns specified
    
    :param analogies_df: analogies test set, of format specified in load_test_data.load_analogy_data
    :param formations: list of which of the 4 words in the quartet to test
    :param topns: set of topN rank cutoffs to assess
    
    :returns: dataframe with columns
        Task
        Theme
        a
        x
        b
        y
        topN_word-X ... for N in topns and for X in [a,x,b,y] - True if word X was in the topN results
        rank_word-X ... for X in [a,x,b,y] - the rank of X in the top 100 results
        avg_topN ... for N in topns - the average of topN_word-X for all X
        avg_rank  - the average of rank_word-X for all X
    """
    analogies_results = [] #analogies_df.copy()
    maxn = max([100] + topns)
    
    order = ['a', 'x', 'b', 'y']
    
    for irow, row in analogies_df.iterrows():
        row_result = {}
        quartet = [row[elem] for elem in order]
        quartet = [word.lower() for word in quartet]
        #display(quartet)
        for i_k in formations:
            try:
                model[quartet[i_k]]
                in_vocab = True
            except KeyError as e:
                in_vocab = False

            if in_vocab:
                try:
                    result = analogy_predict(model, quartet, i=i_k, topn=maxn)
                    failed = False
                except KeyError as e:
                    result = None
                    failed = True
                
                for topn in topns:
                    if not failed:
                        res = result[result.word == quartet[i_k]]
                        if res.shape[0] > 0:
                            ind = res.index[0]
                        else:
                            ind = np.nan
                        
                        row_result[f'top{topn}_word-{order[i_k]}'] = 1 if ind < topn else 0
                        
                    else:
                        row_result[f'top{topn}_word-{order[i_k]}'] = np.nan
                        
                if not failed:
                    row_result[f'rank_word-{order[i_k]}'] = ind + 1
                else:
                    row_result[f'rank_word-{order[i_k]}'] = np.nan
                    
                try:
                    sim_result = analogy_distance(model, quartet, i=i_k)
                    row_result[f'distance_word-{order[i_k]}'] = sim_result.distance.iloc[0]
                except KeyError as e:
                    sim_result = None
                    row_result[f'distance_word-{order[i_k]}'] = np.nan
            else:
                row_result[f'top{topn}_word-{order[i_k]}'] = np.nan
                row_result[f'rank_word-{order[i_k]}'] = np.nan
                row_result[f'distance_word-{order[i_k]}'] = np.nan

            row_result[f'in_vocab-{order[i_k]}'] = 1 if in_vocab else 0

     
        analogies_results.append(row_result)
        
            
    result_df = pd.DataFrame(analogies_results)
    for topn in topns:
        result_df[f'avg_top{topn}'] = result_df.filter(regex=f'top{topn}_').mean(axis=1)
    result_df[f'avg_rank'] = result_df.filter(regex='rank_').mean(axis=1)
    result_df[f'avg_distance'] = result_df.filter(regex='distance_word-').mean(axis=1)
    
    return pd.concat([analogies_df, result_df], axis=1)