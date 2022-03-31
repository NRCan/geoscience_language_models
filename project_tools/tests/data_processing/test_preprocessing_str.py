# Copyright (C) 2021 ServiceNow, Inc.

import pytest
from nrcan_p2.data_processing.preprocessing_str import (
    add_newline,
    sentence_tokenize_spacy_sm,
    sentence_tokenize_spacy_lg,
    tokenize_spacy_sm,
    tokenize_spacy_lg,
    rm_stopwords_spacy,
    rm_punct,
    rm_newline
)

@pytest.mark.parametrize("input_str,expected",
    [
        ("blah \n blah \n", "blah \n blah \n\n"),
        ("blah \n blah ", "blah \n blah \n")
    ]
)
def test_add_newline(input_str, expected):
    result = add_newline(input_str)
    assert result == expected

@pytest.mark.parametrize("input_str,expected",
    [
        ("blah\nblah", "blah blah"),
        ("blah\n", "blah ")
    ]
)
def test_rm_newline(input_str, expected):
    result = rm_newline(input_str)
    assert result == expected    

@pytest.mark.parametrize("input_str,expected",
    [
        ("blah.! blah-! @?-.,:;=()", "blah blah "),
        ("blah\n", "blah\n")
    ]
)
def test_rm_punct(input_str, expected):
    result = rm_punct(input_str)
    assert result == expected        


@pytest.mark.parametrize("input_str, expected",
    [
        ("Here is my sentence. Here is another sentence!", "Here is my sentence.\nHere is another sentence!\n")
    ]
)
def test_sentence_tokenize_spacy_sm(input_str, expected):
    result = sentence_tokenize_spacy_sm(input_str)
    assert result == expected

@pytest.mark.parametrize("input_str, expected",
    [
        ("Here is my sentence. Here is another sentence!", "Here is my sentence.\nHere is another sentence!\n"),
    ]
)
def test_sentence_tokenize_spacy_lg(input_str, expected):
    result = sentence_tokenize_spacy_lg(input_str)
    assert result == expected

@pytest.mark.parametrize("input_str, expected",
    [
        ("""Here. "This," he said, "is ridiculous." His mother-in-law- A.K. Hawings, an old but sharp woman- did not agree...""",
        """Here . " This , " he said , " is ridiculous . " His mother - in - law- A.K. Hawings , an old but sharp woman- did not agree ...""")
    ]
)
def test_tokenize_spacy_sm(input_str, expected):
    result = tokenize_spacy_sm(input_str)
    assert result.strip() == expected.strip()

@pytest.mark.parametrize("input_str, expected",
    [
        ("""Here. "This," he said, "is ridiculous." His mother-in-law- A.K. Hawings, an old but sharp woman- did not agree...""",
        """Here . " This , " he said , " is ridiculous . " His mother - in - law- A.K. Hawings , an old but sharp woman- did not agree ..."""),
        ("""Here omg he said.\n And, then he runn-ing,\n that we didn't do it.""",
        ["""Here omg he said.\n""", """And, then he runn-ing,\n""", """that we didn't do it."""]
        )        
    ]
)
def test_tokenize_spacy_lg(input_str, expected):
    result = tokenize_spacy_lg(input_str)
    assert result.strip() == expected.strip()

@pytest.mark.parametrize("input_str, expected",
    [
        ("""Here omg he said we did n't do it We A.J. Patterson. Do. latterly thence we went the two of us""",
        """omg said A.J. Patterson. Do. went"""),
    ]
)
def test_tokenize_spacy_lg(input_str, expected):
    result = rm_stopwords_spacy(input_str)
    assert result.strip() == expected.strip()    

@pytest.mark.parametrize("input_str, expected",
    [
        ("""Here. "This," he said, "is ridiculous." His mother-in-law- A.K. Hawings, an old but sharp woman- did not agree...
He's gotten away with this again, as a goose get's away with doing anything at all.
I've got him! But how did you get him? Whither did he run?? Didn't you know?
        """,
        # tokenization removes newlines
        """. " , " said , " ridiculous . " mother - - law- A.K. Hawings , old sharp woman- agree ...
 gotten away , goose away .
 got ! ? run ? ? know ?
        """),
        ("Here (he said) don't even try, no.", "( said ) try , .")
    ]
)
def test_tokenize_spacy_lg_rm_stopwords(input_str, expected):
    result = tokenize_spacy_lg(input_str)
    result = rm_stopwords_spacy(result)
    assert result.strip() == expected.strip()    
