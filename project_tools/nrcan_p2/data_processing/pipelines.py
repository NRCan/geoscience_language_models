# Copyright (C) 2021 ServiceNow, Inc.
""" Preprocessing cleaning pipelines """

import nrcan_p2.data_processing.preprocessing_dfcol as preprocessing_dfcol
import nrcan_p2.data_processing.preprocessing_str as preprocessing_str
import nrcan_p2.data_processing.preprocessing_df_filter as preprocessing_df_filter

BASE_PIPELINE = [
    preprocessing_dfcol.rm_dbl_space,
    preprocessing_dfcol.rm_cid,
    preprocessing_dfcol.convert_to_ascii,
    preprocessing_dfcol.rm_nonprintable,
    preprocessing_df_filter.filter_no_letter,
    preprocessing_dfcol.rm_word_all_punct,
    preprocessing_dfcol.rm_newline_hyphenation,
    preprocessing_dfcol.rm_beg_end_word_punct,
    preprocessing_dfcol.rm_punct_mid_punct, 
    preprocessing_dfcol.strip_space,
    preprocessing_df_filter.filter_l2_word,
    preprocessing_df_filter.filter_l4_letter,
    preprocessing_dfcol.rm_mid_word_punct,
    preprocessing_dfcol.rm_non_textual_punct,
    preprocessing_dfcol.rm_newline,
    preprocessing_dfcol.merge_words,
    preprocessing_df_filter.filter_no_real_words_g3letter,
]

# aka V2
BASE_PIPELINE_CLEAN = [
    preprocessing_dfcol.rm_dbl_space,
    preprocessing_dfcol.rm_cid,
    preprocessing_dfcol.convert_to_ascii,
    preprocessing_dfcol.rm_nonprintable,
    preprocessing_df_filter.filter_no_letter,
    preprocessing_dfcol.rm_newline_hyphenation,
    preprocessing_dfcol.rm_newline,    
    preprocessing_df_filter.filter_no_real_words_g3letter, 
    preprocessing_df_filter.filter_with_email,
    preprocessing_dfcol.rm_url,
    preprocessing_dfcol.rm_doi,
    preprocessing_df_filter.filter_with_phonenumber,
    preprocessing_df_filter.filter_non_english,
]

BASE_PIPELINE_PLUS = BASE_PIPELINE_CLEAN + [
    # e.g. "a+b" -> "a + b"
    preprocessing_dfcol.add_space_to_various_punct,
    # remove punct "2+"
    preprocessing_dfcol.squish_punct,
    # remove punct "2 space + "
    preprocessing_dfcol.squish_spaced_punct_no_bracket,
    # drop > 0.1 punct /len
    preprocessing_df_filter.filter_g10_punct,
    # drop < 0.45 real words (don't forget to cap non cap and remove punct)
    preprocessing_df_filter.filter_insufficient_real_words,
    # run merger 
    preprocessing_dfcol.merge_words_2,
    # drop deg 
    preprocessing_dfcol.rm_deg,
]

# do one with >0.8 percent and nothing fancy after filter phonenumbers
BASE_PIPE_80 = BASE_PIPELINE_CLEAN + [
    preprocessing_df_filter.filter_l80_real_words
]

# do one with >0.9
BASE_PIPE_90 = BASE_PIPELINE_CLEAN + [
    preprocessing_df_filter.filter_l90_real_words
]

### GLOVE PREPROCESSING PIPELINES ########################

# V1
SIMPLE_PIPELINE_GLOVE_3 = BASE_PIPELINE + [
    preprocessing_dfcol.tokenize_spacy_lg,
    preprocessing_dfcol.rm_stopwords_spacy,
]

# 80
PIPELINE_GLOVE_80 = BASE_PIPE_80 + [
    preprocessing_dfcol.tokenize_spacy_lg,
    preprocessing_dfcol.rm_stopwords_spacy,
]

# 90
PIPELINE_GLOVE_90 = BASE_PIPE_90 + [
    preprocessing_dfcol.tokenize_spacy_lg,
    preprocessing_dfcol.rm_stopwords_spacy,
]

# PLUS
PIPELINE_GLOVE_PLUS = BASE_PIPELINE_PLUS + [
    preprocessing_dfcol.tokenize_spacy_lg,
    preprocessing_dfcol.rm_stopwords_spacy,
]

### BERT PREPROCESSING PIPELINES ########################

# V1
SIMPLE_PIPELINE_BERT_3 = BASE_PIPELINE

# 80
PIPELINE_BERT_80 = BASE_PIPE_80

# 90
PIPELINE_BERT_90 = BASE_PIPE_90

# PLUS
PIPELINE_BERT_PLUS = BASE_PIPELINE_PLUS

### GLOVE POSTPROCESSING PIPELINES ########################
POSTPIPE_GLOVE = [
    preprocessing_str.rm_punct,     
    preprocessing_str.lower,
    preprocessing_str.rm_newline
]

### BERT POSTPROCESSING PIPELINES ########################
POSTPIPE_BERT_SPACY_2 = [
    preprocessing_str.rm_newline,
    preprocessing_str.sentence_tokenize_spacy_lg,
    preprocessing_str.add_newline,
]