# Copyright (C) 2021 ServiceNow, Inc.
""" Utilities for evaluating word clustering
"""

import numpy as np
import pandas as pd
import itertools 
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import balanced_accuracy_score

def leave_one_out_classification(X,y, clf_init):
    """Perform leave out one cross validation on a classifier 
        initialized using clf_init()
        
    :param X: X data accepted by an sklearn fit
    :param y: y data accepted by an sklearn fit
    :param clf_init: a function with no arguments that can 
        instantiate the potentially multiclass classifier 
        compatable with sklearn 
        e.g. lambda: BernoulliNB()
        
    :returns (preds, scores):
        preds - a list of the actual predictions of the model for each test example
        score - the score (1 for correct, 0 for incorrect) for each test example
    """
    loo = LeaveOneOut()
    preds = []
    scores = []
    for train_index,test_index in loo.split(X,y):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]    
        clf = clf_init()
        clf.fit(X_train,y_train)
        pred = clf.predict(X_test)
        score = clf.score(X_test, y_test)
        preds.append(pred)
        scores.append(score)
    return preds, scores


def get_wordvec_matrix_from_cluster_tests(cluster_tests, model):
    """Collect the wordvectors for a set of words into a matrix suitable for 
        for training, also collect the associated clusters.
        Words that are OOV for the model will not be included in the output
        matrix.
        
    :param cluster_tests: df of the format required by 
        load_test_data.load_word_cluster_data
    :param model: gensim compatable word embedding model
    
    :returns: (X,y) 
        X - the np.array matrix of word vectors for each word/row in cluster_tests
        y - categorical series with the cluster labels
        
    """
    vectors = []
    classes = []
    for irow, row in cluster_tests.iterrows():
        try:
            vector = model[row.Word]
            vectors.append(vector)
            classes.append(row.Cluster)
        except:
            pass
    vectors = np.column_stack(vectors).transpose()
    vectors.shape
    
    classes = pd.DataFrame(classes)[0].astype('category')
    return vectors, classes


def run_loo_classification_for_every_cluster_pair(cluster_tests, model, clf_init):
    """Run leave out one cross validation after training a model on each pair of 
        clusters. 
        
    :param cluster_tests: df of the format required by 
        load_test_data.load_word_cluster_data
    :param model: 
    :param clf_init: no argument function as required by leave_one_out_classification
    
    :returns: all_preds, all_scores, df
        a list of predictions and scores as returned by leave_one_out_classification
        one for each cluster pair
        df - a dataframe with columns [0,1,'scores'] which contains the average 
            balanced accuracy score for each cluster pair's loo classification
    """
    all_cluster_pairs = itertools.combinations(cluster_tests.Cluster.unique(),2)
    all_cluster_pairs = list(all_cluster_pairs)    
    
    all_preds = []
    all_scores = []
    for cluster_pair in all_cluster_pairs:
        print(cluster_pair)
        class_tests = cluster_tests[cluster_tests.Cluster.isin(cluster_pair)]
        X,y = get_wordvec_matrix_from_cluster_tests(class_tests, model)
        preds, scores = leave_one_out_classification(X,y, clf_init)
        all_preds.append(preds)
        all_scores.append(scores)    
        
    ba_scores = []
    for pred, score, cluster_pair in zip(all_preds, all_scores, all_cluster_pairs):
        class_tests = cluster_tests[cluster_tests.Cluster.isin(cluster_pair)]
        X,y = get_wordvec_matrix_from_cluster_tests(class_tests, model)
        ba = balanced_accuracy_score(y, pred)
        ba_scores.append(ba)
    res = pd.DataFrame(all_cluster_pairs)
    res['scores'] = ba_scores
    res    
    
    return all_preds, all_scores, res



