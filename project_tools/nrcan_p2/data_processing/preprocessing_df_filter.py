# Copyright (C) 2021 ServiceNow, Inc.
""" Preprocessing functions, running on df returning df of different sizes """
import re
import string
import langdetect
from nrcan_p2.data_processing.utils import (
    produce_updown_df,
    decide_lang_row
)
from nrcan_p2.data_processing.preprocessing_dfcol import (
    rm_phonenumber
)

PUNCT = re.escape(string.punctuation)
import enchant
EN_DICT = enchant.Dict("en_CA")


def filter_punct_rows_l60(df, col):
    df['punct_count'] = df[col].str.count(f'[{PUNCT}]')
    df['punct_perc'] = df.punct_count / df[col].str.len()
    return df[df.punct_perc < 0.6]


def filter_no_words_g30_letter(df, col):
    df['word_letter_num'] = df[col].str.split().apply(lambda x: [len(re.findall('[a-zA-Z]', xx)) for xx in x])
    df['word_letter_perc'] = df[col].str.split().apply(lambda x: [len(re.findall('[a-zA-Z]', xx))/float(len(xx)) for xx in x])
    return df[df.word_letter_perc.apply(lambda x: any([xx > 0.3 for xx in x]))]


def filter_no_words_g30_letter(df, col):
    df['word_letter_num'] = df[col].str.split().apply(lambda x: [len(re.findall('[a-zA-Z]', xx)) for xx in x])
    df['word_letter_perc'] = df[col].str.split().apply(lambda x: [len(re.findall('[a-zA-Z]', xx))/float(len(xx)) for xx in x])
    return df[df.word_letter_perc.apply(lambda x: any([xx > 0.3 for xx in x]))]    
    

def filter_l20_letter(df, col):
    df['letter_count'] = df[col].str.count(f'[a-zA-Z]')
    df['letter_perc'] = df.letter_count / df[col].str.len()
    return df[df.letter_perc > 0.2]


def filter_l2_word(df, col):
    """ Filter out boxes with < 2 words """
    n_words = df[col].str.split().str.len()
    return df[n_words > 1]    


def filter_l4_letter(df, col):
    """ Filter out boxes with < 4 characters (that are not spaces or newlines)"""
    n_char = df[col].str.strip().str.len() #(f'[a-zA-Z]')
    return df[n_char > 3]        


def filter_no_letter(df, col):
    df['letter_count'] = df[col].str.count(f'[a-zA-Z]')
    return df[df.letter_count > 0]    


def filter_l80_letter(df, col):
    df['letter_count'] = df[col].str.count(f'[a-zA-Z]')
    df['letter_perc'] = df.letter_count / df[col].str.len()
    return df[df.letter_perc > 0.2]    


def filter_no_real_words_g3letter(df, col, en_dict=EN_DICT):
    df['is_enchant_word'] = df[col].str.split().apply(lambda x: [en_dict.check(word) for word in x])
    df['word_char_num'] = df[col].str.split().apply(lambda x: [len(word) for word in x])

    df['is_enchant_word_and_g3l'] = df.apply(lambda row: [is_enchant and (nchar > 3) for is_enchant,nchar in zip(row.is_enchant_word, row.word_char_num)], axis=1)
    df['any_enchant_word_and_g3l'] = df.is_enchant_word_and_g3l.apply(lambda x: any(x))

    return df[df.any_enchant_word_and_g3l]


def filter_lX_perc_words_not_real(df, col, n, en_dict=EN_DICT):
    df['is_enchant_word'] = df[col].str.split().apply(lambda x: [en_dict.check(word) for word in x])
    df['n_enchant_words'] = df.is_enchant_word.apply(lambda x: sum(x))
    df['n_words'] = df[col].str.split().str.len()
    df['perc_enchant_words'] = df.n_enchant_words / df.n_words
    
    return df[df.perc_enchant_words >= n]    


def filter_l80_perc_words_not_real(df, col):
    return filter_lX_perc_words_not_real(df, col, n=0.8)  


def filter_l60_perc_words_not_real(df, col):
    return filter_lX_perc_words_not_real(df, col, n=0.6)


def filter_l50_perc_words_not_real(df, col):
    return filter_lX_perc_words_not_real(df, col, n=0.5)
 

def filter_non_english(df, col, lang_detect_func=langdetect.detect_langs, do_filter=True):
    df['langs'] = df[col].apply(lambda sent: lang_detect_func(sent))
    df['langs_1_prob'] = df['langs'].apply(lambda x: None if len(x) <1 else x[0].prob)
    df['langs_1'] = df['langs'].apply(lambda x: None if len(x) <1 else x[0].lang)
    df = df.drop(columns='langs')

    dff = produce_updown_df(df, 'langs_1') 
    dff['lang'] = dff.apply(lambda row: decide_lang_row(row, 
        text_col=col,
        lang_col='langs_1', 
        lang_prob_col='langs_1_prob',
        lang_up_col='langs_1_up', 
        lang_down_col='langs_1_down', 
    ), axis=1)

    if do_filter:
        return dff[dff.lang == "en"] 
    else:
        return dff


def filter_with_email(df, col):
    """ Filter any boxes that contain emails. These are (almost?) always addresses or disclaimers"""

    df['has_email'] = df[col].str.contains(r'([\w.\-]+@[\w\-.]+[.]([\w\-.]+)?[\w])', regex=True)

    return df[~df.has_email]


def filter_with_phonenumber(df, col):
    df['rm_phonenumber'] = rm_phonenumber(df[col])

    df = df[df.rm_phonenumber == df[col]]
    return df.drop(columns='rm_phonenumber')


def filter_g10_punct(df, col):
    punct = re.escape(string.punctuation)
    num_punct = df[col].str.count(f"[{punct}]")
    num_char = df[col].str.len()
    perc_punct =  num_punct / num_char
    return df[perc_punct <= 0.1]


def filter_insufficient_real_words(df, col, en_dict=EN_DICT):
    punct = re.escape(string.punctuation)

    df['real_words'] = df[col].apply(lambda x: [
        en_dict.check(xx) or 
        en_dict.check(xx[0].upper() + xx[1:]) or 
        en_dict.check(xx[0] + xx[1:].lower()) for xx in re.sub(f"[{punct}]", " ", x).split()])

    df['real_words_n']= df.real_words.apply(lambda x: len([xx for xx in x if xx == True]))
    df['real_words_perc']= df.real_words.apply(lambda x: len([xx for xx in x if xx == True])/len(x))

    df['n_words'] = df.real_words.str.len()

    return df[((df.real_words_perc >= 0.60) & (df.n_words > 50)) | ((df.real_words_perc >= 0.55) & (df.n_words > 25)) | ((df.real_words_perc >= 0.50) & (df.n_words > 10)) | ((df.real_words_perc >= 0.45) & (df.n_words > 0))]


def filter_l90_real_words(df, col, en_dict=EN_DICT):
    punct = re.escape(string.punctuation)

    df['real_words'] = df[col].apply(lambda x: [
        en_dict.check(xx) or 
        en_dict.check(xx[0].upper() + xx[1:]) or 
        en_dict.check(xx[0] + xx[1:].lower()) for xx in re.sub(f"[{punct}]", " ", x).split()])

    df['real_words_n']= df.real_words.apply(lambda x: len([xx for xx in x if xx == True]))
    df['real_words_perc']= df.real_words.apply(lambda x: len([xx for xx in x if xx == True])/len(x))

    df['n_words'] = df.real_words.str.len()

    return df[df.real_words_perc > 0.9]


def filter_l80_real_words(df, col, en_dict=EN_DICT):
    punct = re.escape(string.punctuation)

    df['real_words'] = df[col].apply(lambda x: [
        en_dict.check(xx) or 
        en_dict.check(xx[0].upper() + xx[1:]) or 
        en_dict.check(xx[0] + xx[1:].lower()) for xx in re.sub(f"[{punct}]", " ", x).split()])

    df['real_words_n']= df.real_words.apply(lambda x: len([xx for xx in x if xx == True]))
    df['real_words_perc']= df.real_words.apply(lambda x: len([xx for xx in x if xx == True])/len(x))

    df['n_words'] = df.real_words.str.len()

    return df[df.real_words_perc > 0.8]    

