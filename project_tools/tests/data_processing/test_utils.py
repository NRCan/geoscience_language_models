# Copyright (C) 2021 ServiceNow, Inc.

import pytest
import pandas as pd 
import numpy as np

from nrcan_p2.data_processing.utils import (
    produce_updown_df,
    decide_lang
)


def test_produce_updown_df():

    df = pd.DataFrame({
        'text': ['a', "b", "c", "d", "e"],
        'mycol': [0,1,2,3,4],
        'othercol': ['z','y', 'x', 'v', 'w']
    }, index=[5,6,7,8,9])

    expected = pd.DataFrame({
        'text': ['a', "b", "c", "d", "e"],
        'mycol': [0,1,2,3,4],
        'othercol': ['z','y', 'x', 'v', 'w'],
        #'index_up': [None,5,6,7,8],
        'mycol_up': [None,0,1,2,3],
        #'index_down': [6,7,8,9,None],
        'mycol_down': [1,2,3,4,None]
    }, index=[5,6,7,8,9]).fillna(np.nan)
    expected['mycol'] = expected.mycol.astype('float')
    expected['mycol_up'] = expected.mycol_up.astype('float')
    expected['mycol_down'] = expected.mycol_down.astype('float')
    expected.index = expected.index.astype('float')
    
    output = produce_updown_df(df, 
        col='mycol',
    ) 
        #ol_set=['text', 'mycol', 'othercol'])

    print(output)
    print(expected)

    pd.testing.assert_frame_equal(output, expected)

@pytest.mark.parametrize("text, lang, lang_prob, lang_up, lang_down, expected",[
    ("short", "en", None, None, None, "en"),
    ("super super long french text super super super super super long", "fr", None, None, None, "fr"),
    ("short", "fr", 0.9, None, None, "fr"),
    ("short", "fr", 0.7, "fr", "fr", "fr"),
    ("short", "fr", 0.7, "fr", "it", "fr"),
    ("short", "fr", 0.7, "it", "fr", "fr"),
    ("short", "fr", 0.7, "it", "it", "en"), #assume there's only english and french <-- this might be a bad one
    ("short", "it", 0.7, "fr", "fr", "fr"), #assume there's only english and french
    ("short", "it", 0.7, "en", "en", "en"),
    ("short", "fr", 0.7, "en", "fr", "en"), # this is maybe also a bad idea
    ("short", "fr", 0.7, "fr", "en", "en") # this is maybe also a bad idea
])
def test_decide_lang(text, lang, lang_prob, lang_up, lang_down, expected):
    result = decide_lang(text, lang, lang_prob, lang_up, lang_down)
    assert result == expected