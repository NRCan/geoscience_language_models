# Copyright (C) 2021 ServiceNow, Inc.
""" Utilities for evaluating relatedness

"""
import pandas as pd
import numpy as np

a = 'a'
b = 'x (Words with decreasing similarity ->)'

def produce_all_intra_group_comparisons(relatedness_tests, groupby='Theme'):
    """ Produce all pair-wise intragroup comparisons implied by relatedness_tests
        Compare a->a_sim and a->a_diff where a_sim is provided and a_diff is every other 
        entity in the group specified by the groupby column.
        Note that if a->b and b->c, we assume a--c (not related), unless they are marked 
        as synonyms. In which case a->b and b->c implies a->c
        
    :param relatedness_tests: df, of the format provided by load_test_data.load_similarity_data
    :param groupby: column in the df to groupby. a_diff will be drawn from within a group
    
    :example: 
    $ relatedness_tests
    >    Theme  | a    | x (Words with decreasing similarity ->)  
    > 0  animals| dog  | cat
    > 1  animals| croc | alligator
    > 2  food   | fig  | apple
    
    $ produce_all_intra_group_comparisons(relatedness_tests, groupby='Theme')
    >   Task        | Theme   | Type        | irow | a         | a_sim       | a_diff
    > 0 Relatedness | animals | Intra-theme | 0    | dog       | cat         | alligator
    > 0 Relatedness | animals | Intra-theme | 0    | dog       | cat         | croc
    > 0 Relatedness | animals | Intra-theme | 0    | cat       | dog         | alligator
    > 0 Relatedness | animals | Intra-theme | 0    | cat       | dog         | croc   
    > 1 Relatedness | animals | Intra-theme | 1    | croc      | alligator   | dog
    > 1 Relatedness | animals | Intra-theme | 1    | croc      | alligator   | cat
    > 1 Relatedness | animals | Intra-theme | 1    | alligator | croc        | dog
    > 1 Relatedness | animals | Intra-theme | 1    | alligator | croc        | cat       
    """
    all_intra_group_comparisons = []
    
    def add_rows(other_rows_without_a, a, b, row, irow, all_intra_group_comparisons):
        if other_rows_without_a.shape[0] > 0:

            other_values = other_rows_without_a.l.sum()
            other_values = set([xx for xx in other_values if not pd.isnull(xx)])

            new_comparisons = [{'a': a, 'a_sim': b, 'a_diff': other_value}
                               for other_value in other_values if other_value != b]
            new_comparisons = pd.DataFrame.from_records(new_comparisons)
            new_comparisons['Task'] = row.Task
            new_comparisons['Theme'] = row.Theme
            new_comparisons['Type'] = 'Intra-theme'
            new_comparisons['rowid'] = irow
            all_intra_group_comparisons.append(new_comparisons)        

    for theme_name, theme_group in relatedness_tests.groupby(groupby):
        for irow, row in theme_group.iterrows():
            # find all pairs for the value at a...
            if row['Include synonym']:
                other_rows_without_a = theme_group[(theme_group.a != row.a) & (theme_group[b] != row.a) &
                                                  (theme_group.a != row[b]) & (theme_group[b] != row[b])]
            else:
                other_rows_without_a = theme_group[(theme_group.a != row.a) & (theme_group[b] != row.a)]

            add_rows(other_rows_without_a, row.a, row[b], row, irow, all_intra_group_comparisons)
            
            # find all the pairs for the value at b...
            if row['Include synonym']:
                other_rows_without_a = theme_group[(theme_group.a != row.a) & (theme_group[b] != row.a) &
                                                  (theme_group.a != row[b]) & (theme_group[b] != row[b])]
            else:
                other_rows_without_a = theme_group[(theme_group.a != row[b]) & (theme_group[b] != row[b])]

            add_rows(other_rows_without_a, row[b], row[a], row, irow, all_intra_group_comparisons)

    all_intra_group_comparisons = pd.concat(all_intra_group_comparisons)
    
    return all_intra_group_comparisons
    
    
# Get all outra-theme negative comparisons (a,b to c not in theme)
def produce_all_inter_group_comparisons(relatedness_tests):
    """ Produce all pair-wise intergroup comparisons implied by relatedness_tests
        Compare a->a_sim and a->a_diff where a_sim is provided and a_diff is every other 
        entity NOT in the same group as a (where group is specified by the groupby column).
        
        This tests that a is closer to its partner than it is to anything else in another group.
        Note that it is unclear whether a should be more similar to something in its group that is
        not its partner, vs something in another group.
        
    :param relatedness_tests: df, of the format provided by load_test_data.load_similarity_data
    :param groupby: column in the df to groupby. a_diff will be drawn from within a group
    
    :example:
    $ relatedness_tests
    >    Theme  | a    | x (Words with decreasing similarity ->)  
    > 0  animals| dog  | cat
    > 1  animals| croc | alligator
    > 2  food   | fig  | apple
    
    $ produce_all_intra_group_comparisons(relatedness_tests, groupby='Theme')
    >   Task        | Theme   | Type        | irow | a         | a_sim     | a_diff
    > 0 Relatedness | animals | Outra-theme | 0    | dog       | cat       | fig
    > 0 Relatedness | animals | Outra-theme | 0    | dog       | cat       | apple
    > 0 Relatedness | animals | Outra-theme | 0    | cat       | dog       | fig
    > 0 Relatedness | animals | Outra-theme | 0    | cat       | dog       | apple
    > 1 Relatedness | animals | Outra-theme | 0    | croc      | alligator | fig
    > 1 Relatedness | animals | Outra-theme | 0    | croc      | alligator | apple
    > 1 Relatedness | animals | Outra-theme | 0    | alligator | croc      | fig
    > 1 Relatedness | animals | Outra-theme | 0    | alligator | croc      | apple
    > 2 Relatedness | food    | Outra-theme | 0    | fig       | apple     | dog
    > 2 Relatedness | food    | Outra-theme | 0    | fig       | apple     | cat
    > 2 Relatedness | food    | Outra-theme | 0    | fig       | apple     | croc
    > 2 Relatedness | food    | Outra-theme | 0    | fig       | apple     | alligator    
    > 2 Relatedness | food    | Outra-theme | 0    | apple     | fig       | dog
    > 2 Relatedness | food    | Outra-theme | 0    | apple     | fig       | cat
    > 2 Relatedness | food    | Outra-theme | 0    | apple     | fig       | croc
    > 2 Relatedness | food    | Outra-theme | 0    | apple     | fig       | alligator    
    """    
    all_inter_group_comparisons = []
    
    def add_rows(other_rows_without_a, a, b, row, irow, all_inter_group_comparisons):
        if other_rows_without_a.shape[0] > 0:

            other_values = other_rows_without_a.l.sum()

            other_values = set([xx for xx in other_values if not pd.isnull(xx)])

            new_comparisons = [{'a': a, 'a_sim': b, 'a_diff': other_value}
                               for i, other_value in enumerate(other_values) if other_value != b]
            new_comparisons = pd.DataFrame.from_records(new_comparisons)
            new_comparisons['Task'] = row.Task
            new_comparisons['Theme'] = row.Theme
            new_comparisons['Type'] = 'Outra-theme'
            new_comparisons['rowid'] = irow
            all_inter_group_comparisons.append(new_comparisons)  
            

    for theme_name, theme_group in relatedness_tests.groupby('Theme'):
        # within group random selection
        for irow, row in theme_group.iterrows():
            # get random item within group
            other_rows_without_a = relatedness_tests[(relatedness_tests.a != row.a) & (relatedness_tests[b] != row.a) 
                                                    & (relatedness_tests.Theme != row.Theme)]
            add_rows(other_rows_without_a, row[a], row[b], row, irow, all_inter_group_comparisons)
            #display(other_rows_without_a)


            other_rows_without_a = relatedness_tests[(relatedness_tests.a != row[b]) & (relatedness_tests[b] != row[b]) 
                                                    & (relatedness_tests.Theme != row.Theme)]
            add_rows(other_rows_without_a, row[b], row[a], row, irow, all_inter_group_comparisons)       


    all_inter_group_comparisons = pd.concat(all_inter_group_comparisons)
    return all_inter_group_comparisons

# Get all intra-row comparisons
def produce_intrarow_comparisons(relatedness_tests):
    """ Produce all pair-wise intrarow comparisons implied by relatedness_tests
        Compare a->a_sim and a->a_diff where a_sim is closer to a than a_diff in a gradient relation
        
        E.g. A, B, C, D, E
        
        is provided and a_diff is every other 
        entity NOT in the same group as a (where group is specified by the groupby column).
        
        This tests that a is closer to its partner than it is to anything else in another group.
        Note that it is unclear whether a should be more similar to something in its group that is
        not its partner, vs something in another group.
        
    :param relatedness_tests: df, of the format provided by load_test_data.load_similarity_data
    :param groupby: column in the df to groupby. a_diff will be drawn from within a group
    
    :example: 
    $ relatedness_tests
    >    Theme  | a    | x (Words with decreasing similarity ->) |   |   |   
    > 0  letter | A    | B                                       | C | D | E 
    
    $ produce_all_intra_group_comparisons(relatedness_tests, groupby='Theme')
    >   Task        | Theme   | Type     | irow | a | a_sim | a_diff
    > 0 Relatedness | letter  | Gradient | 0    | A | B     | C
    > 0 Relatedness | letter  | Gradient | 0    | A | B     | D
    > 0 Relatedness | letter  | Gradient | 0    | A | B     | E
    > 0 Relatedness | letter  | Gradient | 0    | A | C     | D
    > 0 Relatedness | letter  | Gradient | 0    | A | C     | E
    > 0 Relatedness | letter  | Gradient | 0    | A | D     | E
    > 0 Relatedness | letter  | Gradient | 0    | B | C     | D    
    > ...
    """
                               
    all_intrarow_comparisons = []

    for irow, row in relatedness_tests[relatedness_tests.l.str.len() > 2].iterrows():
        l = row.l
        for ielem, elem in enumerate(l):
            a = elem

            for iother in range(0,ielem):
                a_diff = l[iother]
                for iiother in range(iother+1, ielem):
                    a_sim = l[iiother]
                    all_intrarow_comparisons.append({
                        'a': elem,
                        'a_sim': a_sim,
                        'a_diff': a_diff,
                        'Task': row.Task,
                        'Theme': row.Theme,       
                        'rowid': irow
                    })

            for iother in range(ielem+1,len(l)):
                a_sim = l[iother]

                for iiother in range(iother+1, len(l)):
                    a_diff = l[iiother]
                    all_intrarow_comparisons.append({
                        'a': elem,
                        'a_sim': a_sim,
                        'a_diff': a_diff,
                        'Task': row.Task,
                        'Theme': row.Theme,
                        'rowid': irow
                    })  


    all_intrarow_comparisons = pd.DataFrame.from_records(all_intrarow_comparisons)
    all_intrarow_comparisons['Type'] = 'Gradient'
    return all_intrarow_comparisons

def produce_all_relatedness_comparisons(relatedness_tests):
    """ Produce all relatedness tests
    
    :param relatedness_tests: df of format specified by load_similarity_data.load_similarity_data
    :returns: df of all relatedness tests with columns
        a a_sim a_diff Task Theme Type rowid
    """
    all_intrarow_comparisons = produce_intrarow_comparisons(relatedness_tests)
    all_inter_group_comparisons = produce_all_inter_group_comparisons(relatedness_tests)
    all_intra_group_comparisons = produce_all_intra_group_comparisons(relatedness_tests)
    
    return pd.concat([all_intrarow_comparisons, all_inter_group_comparisons, all_intra_group_comparisons])


def run_relatedness_comparisons(model, relatedness_tests, to_lower=True):
    """ Produce and run all relatedness tests
    
    :param relatedness_tests: df of format specified by load_similarity_data.load_similarity_data
    :returns: df of all relatedness tests with columns
        a
        a_sim
        a_diff
        Task
        Theme
        rowid
        Type
        Pass - True if a_sim is closer than a_diff, NaN if one of the words in a, a_sim, a_diff is not in the vocab
    """    
    all_comparisons = produce_all_relatedness_comparisons(relatedness_tests)
    
    passed = []
    for irow, row in all_comparisons.iterrows():
        try:
            if to_lower:
                result_sim = model.similarity(row.a.lower(), row.a_sim.lower())
                result_diff = model.similarity(row.a.lower(), row.a_diff.lower())
            else:
                result_sim = model.similarity(row, row.a_sim)
                result_diff = model.similarity(row, row.a_diff)
                
            if result_sim > result_diff:
                passed.append(True)
            else:
                passed.append(False)
        except KeyError as e:
            # Nan if one of the words is not present
            passed.append(np.nan)
            
        
    all_comparisons['Pass'] = passed
    
    return all_comparisons