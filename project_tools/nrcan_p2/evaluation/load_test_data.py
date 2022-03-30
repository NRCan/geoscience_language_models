# Copyright (C) 2021 ServiceNow, Inc.
""" Utilities for loading NRCan specific evaluation test files

"""
import pandas as pd


def load_analogy_data(filename):
    """ Read an analogy csv file of the format
        Task, Theme, a, x, b, y
        
    :param filename: path to csv file
    :returns: df with the above columns
    
    """
    data = pd.read_csv(filename, header=[0])
    assert sorted(list(data.columns)) == sorted(list(['Task', 'Theme', 'a', 'x', 'b', 'y'])), f"Analogy file {filename} does not have expected format"
    
    print(f'Analogy data contains: {data.shape[0]} analogies in {data.groupby("Theme").ngroups} themes')
    
    return data


def load_similarity_data(filename):
    """ Read a similarity dataset of the format
        Task, Theme, Include synonym, Spelling variation, y, a, x (Words with decreasing similarity ->)
        
    :param filename: path to csv file
    :returns: df with the above columns, as well as 
      any extra columns (unnamed)
      l: the values of all columns to the right and including column a, concatenated into a list
    
    """
    column_list = ["Task", "Theme", "Include synonym", "Spelling variation", "y", "a", "x (Words with decreasing similarity ->)"]
    data = pd.read_csv(filename, header=[0])
    cols = sorted(list(data.columns)[0:len(column_list)])
    expected_columns = sorted(list(column_list))
    assert cols == expected_columns, f"Relatedness file {filename} does not have expected format: got {cols}, expected {expected_columns}"

    data['l'] = data.loc[:,'a':].values.tolist()
    data['l'] = data.l.apply(lambda x: [xx for xx in x if xx is not None and not pd.isnull(xx)])    
    
    print(f'Relatedness data contains: {data.shape[0]} relatedness examples in {data.groupby("Theme").ngroups} themes')
    
    return data    


def load_nearest_neighbour_data(filename):
    """ Read a nearest neighbour dataset of the format
        ID, Word
        
    :param filename: path to the csv file
    :returns: df with the above columns
    """
    
    column_list = ['ID', 'Word']
    data = pd.read_csv(filename, header=[0])
    cols = sorted(list(data.columns))
    assert cols == sorted(list(column_list))
                          
    print(f'Nearest neighbor data contains: {data.shape[0]} words')
    return data
    
    
def load_word_cluster_data(filename):
    """ Read a word cluster dataset of the format
        Cluster, Word
        
    :param filename: path to the csv
    :returns: df with the above columns
    """
    
    column_list = ['Cluster', 'Word']
    data = pd.read_csv(filename, header=[0])
    cols = sorted(list(data.columns))
    assert cols == sorted(list(column_list))
    
    print(f'Word cluster data contains: {data.groupby("Cluster").ngroups} clusters and {data.shape[0]} total words')
    return data