# Copyright (C) 2021 ServiceNow, Inc.
""" Preprocessing functions, running on text inputs """

import re
import spacy 
from spacy.lang.en.stop_words import STOP_WORDS
import string


SPACY_MODEL_SM = spacy.load('en_core_web_sm')
SPACY_MODEL_LG = spacy.load('en_core_web_lg')


def sentence_tokenize_spacy_sm(text: str):
    """ Run spacy en_core_web_sm tokenization on text, combining each sentence with newlines
    (Each sentence, including the last one will terminate in a newline.)
    """
    spacy_model = SPACY_MODEL_SM
    return "\n".join([x.text for x in spacy_model(text).sents]) + "\n"
    #return df_col.apply(lambda x: spacy_model(x).sents)


def sentence_tokenize_spacy_lg(text: str):
    """ Run spacy en_core_web_sm tokenization on text, combining each sentence with newlines
    (Each sentence, including the last one will terminate in a newline.)
    """
    spacy_model = SPACY_MODEL_LG
    spacy_model.max_length = 10000000 

    return "\n".join([x.text for x in spacy_model(text, disable = ['ner']).sents]) + "\n"
    #return df_col.apply(lambda x: spacy_model(x).sents)


def add_newline(text: str):
    return text + "\n"


def rm_newline(text: str):
    return re.sub("\s+", " ", re.sub("\n", " ", text))


def rm_punct(text: str):
    punct = re.escape(string.punctuation)
    text = re.sub(f"[{punct}]", " ", text)
    text = re.sub(f" +", " ", text)
    return text


def lower(text: str):
    return text.lower()


def rm_stopwords_spacy(text, stop_words=STOP_WORDS):
    for word in stop_words:
        reg = re.compile(r'(?:(?<=\s)|(?<=^))' + word + r'(?:(?=\s)|(?=$))', re.IGNORECASE)
        text =  reg.sub(' ', text)
    text = re.sub(" +", " ", text)

    return text    


def add_docend(text):
    return text + "[DOCEND]"


def tokenize_spacy_sm(text):
    spacy_model = SPACY_MODEL_SM
    
    return ' '.join([token.text for token in spacy_model(text)])


def tokenize_spacy_lg(text):
    spacy_model = SPACY_MODEL_LG
    spacy_model.max_length = 1500000 

    text_lines = text.split('\n')
    new_lines = []
    for line in text_lines:
        new_line = ' '.join([token.text for token in spacy_model(line, disable = ['ner', 'parser'])])
        new_lines.append(new_line)
    return '\n'.join(new_lines)
