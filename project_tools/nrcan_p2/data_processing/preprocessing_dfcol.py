# Copyright (C) 2021 ServiceNow, Inc.
""" Preprocessing functions running on df str column inputs 
    (optimized for df computation)

    These functions will **fail** if the string column contains null values
"""

from unidecode import unidecode, unidecode_expect_nonascii, unidecode_expect_ascii
import re
import numpy as np
import string
import enchant
import nrcan_p2.data_processing.preprocessing_str as preprocessing_str
EN_DICT = enchant.Dict("en_CA")

SPECIAL_CASES = {
    'ofthe': 'of the',
    'inthe': 'in the',
    'forthe': 'for the',
    'onthe': 'on the'
}

SPECIAL_CASES_2 = {
    'ofthe': 'of the',
    'inthe': 'in the',
    'forthe': 'for the',
    'onthe': 'on the',
    "andiron": "and iron"
}

def rm_dbl_space(dfcol):
    """ Reduce multiple whitespace to single space (do not touch newlines) """
    return dfcol.str.replace(r'[^\S\n]+', ' ', regex=True)


def rm_cid(dfcol):
    """ Remove (cid:X) where X is a number """
    return dfcol.str.replace(r'\(cid:[0-9]+\)', '', regex=True)


def strip_space(dfcol):
    """ Strip spaces (not newline) """
    return dfcol.str.strip(r' ')


def rm_dbl_punct(dfcol):
    """ Remove doubled punctuation characters (the same character, repeated) 
        Except for period, which we allow to exist 3 or more times.
    """
    # everything except .
    s = r"([!\-\"#$%&\'()*+,/:;<=>?@[\\\]^_`{|}~])\1+"
    # period 
    ss = r"([.])\1{3,}"    
    return dfcol.str.replace(s, r'\1', regex=True).str.replace(ss, r'\1', regex=True)


def rm_word_all_punct(dfcol):
    """ Remove words that are entirely punctuation """
    punct = re.escape(string.punctuation) 
    ss = f"((?<=\s)|^)([{punct}]+)((?=\s)|$)"
    return dfcol.str.replace(ss, r'', regex=True) 

    
def convert_to_ascii(dfcol):
    """ Convert non-ascii characters to their ascii equivalent if it exists """
    # failures in the unidecode function replace characters with [?], which we must replace
    return dfcol.apply(lambda x: unidecode_expect_ascii(x)).str.replace('\[\?\]', ' ', regex=True)


def lower(dfcol):
    """ Lowercase """
    return dfcol.str.lower()


def rm_newline(dfcol):
    """ Remove newlines (also re-removing any double spaces)"""
    return rm_dbl_space(dfcol.str.replace(r"\n", " ", regex=True))


def rm_newline_except_end(dfcol):
    """ Remove newlines, except those at string end 
        This function is useful for debugging. 
    """
    return rm_dbl_space(dfcol.str.replace(r"\n\s*(?!$)", " ", regex=True))    


def rm_nonprintable(dfcol):
    """ Remove all non printable characters """
    remove_printables_str = f'[^{re.escape(string.printable)}]'
    return rm_dbl_space(dfcol.str.replace(remove_printables_str, ' ', regex=True))


def rm_punct(dfcol):
    return rm_dbl_space(dfcol.str.replace(r'[^\w\s]|_',' ', regex=True))


def rm_mid_word_punct(dfcol):
    """ Aggressively remove punctuation mid-word """
    punct = re.escape(string.punctuation)
    mid_text_illegal_punct = re.escape('!"#$%&\()*+,/:;<=>?@[\\]^_`{|}~') #text punctuation - , ' or .
    rstr = f'(?<=[a-zA-Z0-9]|[{punct}])[{mid_text_illegal_punct}]+(?=[a-zA-Z0-9]|[{punct}])'
    rstr2 = f'((?<=[a-zA-Z0-9][{punct}])[{punct}]+(?=[a-zA-Z0-9]))|((?<=[a-zA-Z0-9])[{punct}]+(?=[{punct}][a-zA-Z0-9]))'

    col = dfcol.str.replace(rstr, '', regex=True)
    col = col.str.replace(rstr2, '', regex=True)
    return col


def rm_punct_mid_punct(dfcol):
    """ Remove punctuation that is bordered by other punctuation (even if there's a space in between)
        The case this removes is the .,", which it converts to ." 
    """
    punct = re.escape(string.punctuation)   
    ss = f"(?<=[{punct}])([{punct}]+)(?=[{punct}])"
    return rm_dbl_space(dfcol.str.replace(ss, r'', regex=True))


def rm_mid_word_punct(dfcol):
    punct = re.escape(string.punctuation)
    mid_text_illegal_punct = re.escape('!"#$%&\()*+,/:;<=>?@[\\]^_`{|}~') #text punctuation - no dash ' or .
    rstr = f'(?<=[a-zA-Z]|[{punct}])[{mid_text_illegal_punct}]+(?=[a-zA-Z]|[{punct}])' # (a|?)(?+)(a|?)
    rstr = f'(?<=[a-zA-Z]|[{punct}])[{mid_text_illegal_punct}]+(?=[a-zA-Z]|[{punct}])' # (a|?)(?+)(a|?)

    col = dfcol.str.replace(rstr, '', regex=True)
    return col    

def rm_email(dfcol):
    return rm_dbl_space(dfcol.str.replace(r'([\w.\-]+@[\w\-.]+[.]([\w\-.]+)?[\w])', ' ', regex=True))
    
def rm_doi(dfcol):
    # remove url before running this one
    return rm_dbl_space(dfcol.str.replace(r'(((doi.?:?\s?)|(doi\.org/)|(https://doi\.org/))\s*)(10\.([A-Za-z0-9.\/-]+)?[A-Za-z0-9\/])', ' ', regex=True))

def rm_url(dfcol):
    # run this after removing newline hyphenation (urls often break across pages)
    return rm_dbl_space(dfcol.str.replace(
        r'\b((http(s)?|ftp):\/\/)?(www\.)?(([-a-zA-Z0-9@:%_\+~#=]+\.){1,256})[a-z]{2,6}\b(([-a-zA-Z0-9@:%_\+~#?&//=.]+[-a-zA-Z0-9@%_\+~#?&//=])|[-a-zA-Z0-9@%_\+~#?&//=])?', ' ', regex=True))
  
def rm_phonenumber(dfcol):
    def replace_func(matchobj):
        g0 = matchobj.group(0)
        #print(g0)
        if re.search(r'^\D?\s?\d{3,4}[- ]\d{4}$', g0.strip()):
            return g0
        if not re.search(r"[-+ ]", g0.strip()):
            return g0
        if g0.strip()[0] == "-":
            return g0
        else:
            return ' '

    return rm_dbl_space(dfcol.str.replace(r'(?:(\b|\s|^))(?<![-])(\+?\d{1,2} ?)?1?[-. ]?((\(\d{3}\))|(\d{3}))?[ .-]?\d{3}[ .-]?\d{4}(?:(\b|\s|$))', replace_func, regex=True))


def rm_8d_code_no_dash(dfcol):
    return rm_dbl_space(dfcol.str.replace(r'([0-9A-Z]+[A-Z][0-9A-Z]+[0-9][0-9A-Z]{4,})|([0-9A-Z]+[0-9][0-9A-Z]+[A-Z][0-9A-Z]{4,})', regex=True))
    

def rm_beg_end_word_punct(dfcol):
    """ Remove illegal punctuation from the beginning and end of words 
        Only ").,?!;: may exist at the end of a word
        Only "( may exist at the beginning
        Note that "word" here means letter (numbers will be unaffected).
        Multiple such punctuation will be removed.
    """
    punct = re.escape(string.punctuation)
    beg_illegal_punct = re.escape('!.,#$%&)\*+,/:;<=>-?@\'[\\]^_`{|}~')
    end_illegal_punct = re.escape('#$%&(\*+/<=>@[\\]\'-^_`{|}~')
    rstr2 = f'((?=\s|^)[{beg_illegal_punct}]+(?=[a-zA-Z]|[{punct}])|(?<=[a-zA-Z]|[{punct}])[{end_illegal_punct}]+(?=\s|$))'
    return dfcol.str.replace(rstr2, '', regex=True)


def sep_brackets(dfcol):
    return rm_dbl_space(dfcol.str.replace(r'(\([^\)]+\))', ' \\1 ', regex=True))


def rm_mid_num_punct(dfcol):
    mid_num_punct = re.escape("""!"#$%&'()*+,-/:;<=>?@[\]^_`{|}~""") 
    mid_num_punct_r = f'(((?<=[0-9])(\+\-|\-\+)(?=[0-9]))|((?<=[0-9])[{mid_num_punct}](?=[0-9])))'

    mid_num_punct2 = re.escape(""",""") 
    mid_num_punct2_r = f'(((?<=[0-9])(\+\-|\-\+)(?=[0-9]))|((?<=[0-9])[{mid_num_punct}](?=[0-9])))'

    col = rm_dbl_space(dfcol.str.replace(mid_num_punct_r, ' \\1 ', regex=True))
    return rm_dbl_space(col.str.replace(mid_num_punct2_r, '\\1 ', regex=True))


def rm_triple_chars(dfcol):
    return rm_dbl_space(dfcol.str.replace(r'((.)\2{2,})', r'\2', regex=True))


def rm_non_textual_punct(dfcol):
    """ Aggressively remove almost all punctuation (except .,?:;- ) """
    text_punct = '.,?:;-' #this also removes + <> $ %
    nontext_punct = [char for char in string.punctuation if char not in text_punct]
    nontext_punct = re.escape(''.join(nontext_punct))
    return dfcol.str.replace(f'[{nontext_punct}]', '', regex=True)


def rm_newline_hyphenation(dfcol):
    """ Remove hyphens at the end of lines (continuation character) and merge the text """
    return dfcol.str.replace('([a-z])(-\s*\n\s*)([a-z])', r'\1\3', regex=True)


def merge_words(dfcol, en_dict=EN_DICT):
    """ Merge a words with extra whitespace """
    res = dfcol.str.split().apply(lambda x: compute_best_joining(x, en_dict))     
    res = res.str.join(' ')
    return res

def merge_words_2(dfcol, en_dict=EN_DICT, special_cases=SPECIAL_CASES_2):
    """ Merge a words with extra whitespace """
    res = dfcol.str.split().apply(lambda x: compute_best_joining(x, en_dict))     
    res = res.str.join(' ')
    return res    


def merge_words_bkwd(dfcol, en_dict=EN_DICT):
    """ Merge a words with extra whitespace (backward algorithm) """
    res = dfcol.str.split().apply(lambda x: compute_best_joining_bkwd_recursive(x, en_dict))     
    res = res.str.join(' ')
    return res    


def compute_best_joining_recursive(s_split, en_dict, special_cases=SPECIAL_CASES):
    """ Merge a set of words with extra whitespace (recursive version)
        e.g 'th e on ly w ay' -> 'the only way'

    """

    if len(s_split) == 0:
        return []

    if len(s_split) == 1:
        return [s_split[0]]

    # compute the longest legal word combo starting with word_0
    new_first_word = s_split[0]
    i = 1

    # consider the first word to always be legal (just incase nothing else is)
    legal_first_words = [(s_split[0], 0)]

    while i < len(s_split) and len(new_first_word) < 20: 
        new_first_word += s_split[i]
        if en_dict.check(new_first_word):
            legal_first_words.append((new_first_word, i))
            
        if new_first_word in special_cases:
            found_item = [legal_tuple for legal_tuple in legal_first_words if legal_tuple[0] == special_cases[new_first_word].split()[0]]
            if len(found_item) > 0:
                found_item = found_item[0]
            else:
                # something weird happened (E.g. o nt he -> on the)
                found_item = (special_cases[new_first_word], i)

            legal_first_words.append(found_item)

        i += 1

    # [(a, 0), (and, 1)]
    longest_legal_word, index_of_last_captured = legal_first_words[-1]

    # repeat for the next segment
    subsequent_legal_list = compute_best_joining_recursive(s_split[index_of_last_captured + 1:], en_dict)
    retval = [longest_legal_word] + subsequent_legal_list

    return retval


def compute_best_joining(s_split, en_dict):
    """ Merge a set of words with extra whitespace (iterative version)
        e.g 'th e on ly w ay' -> 'the only way'

    """
    retval = []
    
    while len(s_split) >= 1: 

        if len(s_split) == 1:
            retval.append(s_split[0])
            break

        special_cases = {
            'ofthe': 'of the',
            'inthe': 'in the',
            'forthe': 'for the',
            'onthe': 'on the',
        }

        # compute the longest legal word combo starting with word_0
        new_first_word = s_split[0]
        i = 1

        # consider the first word to always be legal (just incase nothing else is)
        legal_first_words = [(s_split[0], 0)]


        while i < len(s_split) and len(new_first_word) < 20: 
            new_first_word += s_split[i]
            if en_dict.check(new_first_word):
                legal_first_words.append((new_first_word, i))
                
            if new_first_word in special_cases:
                found_item = [legal_tuple for legal_tuple in legal_first_words if legal_tuple[0] == special_cases[new_first_word].split()[0]]
                if len(found_item) > 0:
                    found_item = found_item[0]
                else:
                    # something weird happened (E.g. o nt he -> on the)
                    found_item = (special_cases[new_first_word], i)

                legal_first_words.append(found_item)

            i += 1

        # [(a, 0), (and, 1)]
        longest_legal_word, index_of_last_captured = legal_first_words[-1]

        # repeat for the next segment
        s_split = s_split[index_of_last_captured + 1:]
        retval.append(longest_legal_word)

    return retval    


def compute_best_joining_bkwd_recursive(s_split, en_dict):
    """ Merge a set of words with extra whitespace (recursive version, backward algorithm)
        e.g 'th e on ly w ay' -> 'the only way'

    """    
    if len(s_split) == 0:
        return []

    if len(s_split) == 1:
        return [s_split[0]]


    # compute the longest legal word combo starting with word_0
    new_first_word = s_split[-1]
    i = len(s_split) - 2

    # consider the first word to always be legal (just incase nothing else is)
    legal_first_words = [(s_split[-1], len(s_split)-1)]
    while i >= 0  and len(new_first_word) < 20: 
        #old_new_first_word = new_first_word
        new_first_word = s_split[i] + new_first_word
        if en_dict.check(new_first_word):
            legal_first_words.append((new_first_word, i))
        i -=1

    # [(a, 0), (and, 1)]
    longest_legal_word, index_of_last_captured = legal_first_words[-1]

    # repeat for the next segment
    subsequent_legal_list = compute_best_joining_bkwd_recursive(s_split[0: index_of_last_captured:], en_dict)
    retval = subsequent_legal_list + [longest_legal_word]

    return retval    


def rm_stopwords_spacy(dfcol):
    from spacy.lang.en.stop_words import STOP_WORDS
    col = dfcol
    for word in STOP_WORDS:
        col = col.str.replace(f'\b{word}\b', '')
    return col


def tokenize_spacy_lg(dfcol):
    return dfcol.apply(preprocessing_str.tokenize_spacy_lg)


def rm_stopwords_spacy(dfcol):
    return dfcol.apply(preprocessing_str.rm_stopwords_spacy)


def rm_slash(dfcol):
    return rm_dbl_space(dfcol.str.replace(r'/', ' ', regex=True))


def rm_hyphen(dfcol):
    return rm_dbl_space(dfcol.str.replace(r'[-]', ' ', regex=True))


def add_space_to_bracket(dfcol):
    col = dfcol.str.replace(r'\(', ' ( ', regex=True)
    col = col.replace(r'\)', ' ) ', regex=True)
    return rm_dbl_space(col)


def squish_punct(dfcol):
    def replacement_func(matchobj):
        g0 = matchobj.group(0)
        g2 = matchobj.group(1)
        
        if len(g0) < 5:
            if g0 in [".,", "(\")", "\")", ",\"", ";\"",  "\",", "\"." ".,", "),", ").", ".)", "%;", "%.", "%,", ".\"" "\";", "):", "?)", "%:"]:
                return g0

            if g0 in ["(?)", "(%)"]:
                return g0
        
        if len(g0.split()) == 0:
            return ""
        
        m0 = g0.split()[0]
        
        if m0[-1] in [".", ",", ";"]:
            return m0[-1]

        return g0[0]    

    punct = re.escape(string.punctuation)
    ms = f"([{punct}]" + "{2,}" + f")"

    return rm_dbl_space(dfcol.str.replace(ms, replacement_func, regex=True))


def squish_spaced_punct_no_bracket(dfcol):
    def replacement_func(matchobj):
        g0 = matchobj.group(0)
        g2 = matchobj.group(1)

        return g2[0] + " "

    ms = r"(([!\"\#\$%\&'\*\+,\-\./:;<=>\?@\[\\\]\^_`\{\|\}\~]\s+){3,})"
    return rm_dbl_space(dfcol.str.replace(ms, replacement_func, regex=True))


def add_space_to_various_punct(dfcol):

    return rm_dbl_space(dfcol.str.replace(r"([+=\[\]\(\)\/\-*:])", ' \\1 '))


def rm_deg(dfcol):
    return rm_dbl_space(dfcol.str.replace(r"\S*[0-9]deg[0-9NSEW]\S*", " "))
