# Copyright (C) 2021 ServiceNow, Inc.

import pandas as pd
import pytest
from nrcan_p2.data_processing.preprocessing_df_filter import (
    filter_l2_word,
    filter_l4_letter,
    filter_no_real_words_g3letter,
    filter_non_english,
    filter_with_email
)


@pytest.mark.parametrize("text_col, expected_text",
    [
        (['I am a word', 'Word', ' ..  ', 'Boo boo', 'I am too'],
         ['I am a word', 'Boo boo', 'I am too']
        )
    ]
)
def test_filter_l2_word(text_col, expected_text):
    df_test = pd.DataFrame({'text': text_col})
    res = filter_l2_word(df_test, 'text')
    expected_df = pd.DataFrame({'text': expected_text})
    res = res.reset_index(drop=True)
    pd.testing.assert_frame_equal(res, expected_df) 


@pytest.mark.parametrize("text_col, expected_text",
    [
        (['    ', 'Wor', '..', 'Booo', 'I am too'],
         ['Booo', 'I am too']
        )
    ]
)
def test_filter_l4_letter(text_col, expected_text):
    df_test = pd.DataFrame({'text': text_col})
    res = filter_l4_letter(df_test, 'text')
    expected_df = pd.DataFrame({'text': expected_text})
    res = res.reset_index(drop=True)
    pd.testing.assert_frame_equal(res, expected_df)        


@pytest.mark.parametrize("text_col, expected_text",
    [
        (['I am to sm', 'Amazing', 'biui jakshfajs lkjskljf aslkjjl', 'I am a real'],
         ['Amazing', 'I am a real']
        )
    ]
)
def test_filter_no_real_words_g3letter(text_col, expected_text):
    df_test = pd.DataFrame({'text': text_col})
    res = filter_no_real_words_g3letter(df_test, 'text')
    expected_df = pd.DataFrame({'text': expected_text})
    res = res.reset_index(drop=True)
    res = res[['text']]
    expected_df = expected_df[['text']]
    pd.testing.assert_frame_equal(res, expected_df)   

@pytest.mark.parametrize("text_col",[
    (["This is clearly an english sentence",
      "et al.", #keep this because it's sandiched bewteen english
      "This is also english",
      "Mais ceci est clairement pas anglais", # high prob, keep it
      "et ceci aussi", #
      "et ceci",
      "et ceci",
      "English", "english", "francais", "francais", 
      "questo e chiaramente una parola italiano"])
])
def test_filter_non_english(text_col):
    df_test = pd.DataFrame({'text': text_col})       

    res = filter_non_english(df_test, col='text', do_filter=False)
    print(res)
    pd.testing.assert_frame_equal(res, None)


@pytest.mark.parametrize("text_col, expected_text", [
    (["This a.j.hopkins@hopkin.hop.com name",
      "crazy@crazy-crazy.crazy-crazy.org",
      "I have no emails",
      "This a.j.hopkins@hopkin.hop.com. This name",
      "This (crazy-@j.com) there",
      "I am also email free"
    ],
    ["I have no emails",
      "I am also email free"]),
])
def test_rm_email(text_col, expected_text):
    df_test = pd.DataFrame({'text': text_col})
    res = filter_with_email(df_test, 'text')
    expected_df = pd.DataFrame({'text': expected_text})
    res = res[['text']].reset_index(drop=True)
    pd.testing.assert_frame_equal(res, expected_df)          

