# Copyright (C) 2021 ServiceNow, Inc.
""" Utility functions """

import pandas as pd


def decide_lang_row(row, text_col:str, lang_col:str, lang_prob_col:str, lang_up_col:str, lang_down_col:str):
    """ Decide the language of a row of text in a dataframe 
        It requires the following columns to be *already present* in the dataframe:
        * text_col - the text
        * lang_col - language of the text as a string (e.g. "en" or "fr") as output by langdetect
        * lang_prob_col - the 
    """
    return decide_lang(
        text=row[text_col],
        lang=row[lang_col],
        lang_prob=row[lang_prob_col],
        lang_up=row[lang_up_col],
        lang_down=row[lang_down_col]
    )


def decide_lang(text:str, lang:str, lang_prob:float, lang_up:str, lang_down:str):
    """ Logic for determining the language of a piece of text, based on the 
        language of the previous and subsequent texts. 

    :param text: the text string
    :param lang: the language of the text (e.g. "en" or "fr" as output by langdetect)
    :param lang_prob: the probability of the text's language
    :param lang_up: the language of the previous text
    :param lang_down: the language of the subsequent text
    """
    if lang == "en":
        # trust that en were classified correctly...
        return "en"

    if lang == "fr":
        # for french, we have to be careful that it doesn't ditch real (necessary) english text

        if len(text) > 50:
            # if it's long, trust it
            return "fr"

        if lang_prob >= 0.9:
            # if it's high prob, trust it
            return "fr"

        if lang_up == "fr" and lang_down == "fr":
            # if both the previous and next row are also "fr"
            return "fr"

        if lang_up == "fr" and lang_down != "en" or lang_up != "en" and lang_down == "fr":
            return "fr"
        
        if lang_up != "fr" and lang_down != "fr":
            return "en"

    if not lang in ['en', 'fr']:
        if lang_up == "fr" and lang_down == "fr":
            return "fr"
        
        if lang_up == "en" and lang_down == "en":
            return "en"       
        
    return "en"   


def produce_updown_df_grouped(df, col, colset, groupbycol):
    """ Produce a new df with the two new columns containing the 
        value of the column col for the previous and subsequent rows.
        Previous and subsequent rows are limited to within unique 
        groups of the column groupbycol
    """
    overall_df = []

    for gname, group in df.groupby(groupbycol):
        overall_df.append(produce_updown_df(group, col))
    
    return overall_df


def produce_updown_df(df, col):
    """ Return a df with two new columns named:
        * {col}_up
        * {col}_down
        which have the same value as {col} column, but
        for the row above and the row below in the original dataframe 

        Note: integer columns will be converted to floats 
        columns 

    """
    dfs_up = df[[col]].copy()
    dfs_up = dfs_up.reset_index()
    dfs_up.index = dfs_up.index +1
    dfs_up.columns = [f"{col}_up" for col in dfs_up.columns]

    dfs_down = df[[col]].copy()
    dfs_down = dfs_down.reset_index()
    dfs_down.index = dfs_down.index - 1
    dfs_down.columns = [f"{col}_down" for col in dfs_down.columns]

    df1 = pd.concat([df.reset_index(), dfs_up, dfs_down], axis=1).iloc[1:-1]
    df1 = df1.set_index('index')
    df1 = df1.drop(columns=['index_up', 'index_down'])
    df1.index.name = None
    return df1