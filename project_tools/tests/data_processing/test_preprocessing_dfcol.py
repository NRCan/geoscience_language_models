# Copyright (C) 2021 ServiceNow, Inc.

import pytest
import pandas as pd
import re
from nrcan_p2.data_processing.preprocessing_dfcol import (
    rm_dbl_space,
    rm_cid,
    rm_dbl_punct,
    convert_to_ascii,
    lower,
    rm_punct,
    rm_newline,
    rm_triple_chars,
    rm_mid_num_punct,
    rm_word_all_punct,
    rm_newline_hyphenation,
    rm_mid_word_punct,
    rm_beg_end_word_punct,
    merge_words,
    merge_words_bkwd,
    rm_nonprintable,
    rm_punct_mid_punct,
    rm_non_textual_punct,
    rm_newline_except_end,
    strip_space,
    rm_email,
    rm_url,
    rm_doi,
    rm_phonenumber,
    rm_slash
)

@pytest.mark.parametrize("text_col, expected_text",
    [
        (['Alaska. \n', '   blah  \t\t \t \n blah'],
         ['Alaska. \n', ' blah \n blah']
        )
    ]
)
def test_rm_dbl_space(text_col, expected_text):
    df_test = pd.DataFrame({'text': text_col})
    res = rm_dbl_space(df_test.text)
    assert list(res.values) == expected_text  


@pytest.mark.parametrize("text_col, expected_text",
    [
        (['Alaska. \n', '(cid:1010)blah(cid:4)\n'],
         ['Alaska. \n', 'blah\n']
        )
    ]
)
def test_rm_cid(text_col, expected_text):
    df_test = pd.DataFrame({'text': text_col})
    res = rm_cid(df_test.text)
    assert list(res.values) == expected_text  


@pytest.mark.parametrize("text_col, expected_text",
    [
        (['  Alaska. \n', '\nblah  \n  '],
         ['Alaska. \n', '\nblah  \n']
        )
    ]
)
def test_strip_space(text_col, expected_text):
    df_test = pd.DataFrame({'text': text_col})
    res = strip_space(df_test.text)
    assert list(res.values) == expected_text  


@pytest.mark.parametrize("text_col, expected_text",
    [
        (['Alaska. \n', '||||kkll-ll???!!??...??....'],
         ['Alaska. \n', '|kkll-ll?!?...?.']
        )
    ]
)
def test_convert_to_ascii(text_col, expected_text):
    df_test = pd.DataFrame({'text': text_col})
    print(df_test)
    res = rm_dbl_punct(df_test.text)
    assert list(res.values) == expected_text   


@pytest.mark.parametrize("text_col, expected_text",
    [
        (['Alaska. \n', '𝟏−𝑨𝑨𝑹.. \n', "1  %>+* .B 4!\".𝐵 "],
         ['Alaska. \n', '1-AAR.. \n',  "1  %>+* .B 4!\".B "]
        )
    ]
)
def test_convert_to_ascii(text_col, expected_text):
    df_test = pd.DataFrame({'text': text_col})
    res = convert_to_ascii(df_test.text)
    assert list(res.values) == expected_text   


@pytest.mark.parametrize("text_col, expected_text",
    [
        (['Alaska. \n', 'AL_aska.. \n'],
         ['alaska. \n', 'al_aska.. \n']
        )
    ]
)
def test_lower(text_col, expected_text):
    df_test = pd.DataFrame({'text': text_col})
    res = lower(df_test.text)
    assert list(res.values) == expected_text   


@pytest.mark.parametrize("text_col, expected_text",
    [
        (['Alaska. \n', 'Al_aska.. \n'],
         ['Alaska \n', 'Al aska \n']
        )
    ]
)
def test_rm_punct(text_col, expected_text):
    df_test = pd.DataFrame({'text': text_col})
    print(df_test)
    res = rm_punct(df_test.text)
    assert list(res.values) == expected_text    


@pytest.mark.parametrize("text_col, expected_text",
    [
        (['Alaska. \n\n', '\nAl_aska.. \n'],
         ['Alaska. ', ' Al_aska.. ']
        )
    ]
)
def test_rm_newline(text_col, expected_text):
    df_test = pd.DataFrame({'text': text_col})
    res = rm_newline(df_test.text)
    assert list(res.values) == expected_text  


@pytest.mark.parametrize("text_col, expected_text",
    [
        (['..!. ..a. @.@ .. !!', 'Thhh.!.iiiiiiiss ! ~ is bad...', '"This!"', '"This.,"'],
         ['.. ..a. @@ .. !!', 'Thhh..iiiiiiiss ! ~ is bad..', '"This!"', '"This."']
        )
    ]
)
def test_rm_punct_mid_punct(text_col, expected_text):
    df_test = pd.DataFrame({'text': text_col})
    res = rm_punct_mid_punct(df_test.text)
    assert list(res.values) == expected_text  


@pytest.mark.parametrize("text_col, expected_text",
    [
        (['..!. ..a. @.@ .. !!', 'Thhhiiiiiiiss ! ~ is bad...'],
         [' ..a.   ', 'Thhhiiiiiiiss   is bad...']
         
        )
    ]
)
def test_rm_word_all_punct(text_col, expected_text):
    df_test = pd.DataFrame({'text': text_col})
    res = rm_word_all_punct(df_test.text)
    assert list(res.values) == expected_text  


@pytest.mark.parametrize("text_col, expected_text",
    [
        (['..\n\n !!\n', '\nThhhii \n    '],
         ['.. !!\n', ' Thhhii ']
         
        )
    ]
)
def test_rm_newline_except_end(text_col, expected_text):
    df_test = pd.DataFrame({'text': text_col})
    res = rm_newline_except_end(df_test.text)
    assert list(res.values) == expected_text  



@pytest.mark.parametrize("text_col, expected_text",
    [
        (['This is normal..', 'Thhhiiiiiiiss is bad...'],
         ['This is normal..', 'Thiss is bad.']
        )
    ]
)
def test_rm_triple_chars(text_col, expected_text):
    df_test = pd.DataFrame({'text': text_col})
    res = rm_triple_chars(df_test.text)
    assert list(res.values) == expected_text  


@pytest.mark.parametrize("text_col, expected_text",
    [
        (['This is normal..', 'This\x07 is bad \x0f'],
         ['This is normal..', 'This is bad ']
        )
    ]
)
def test_rm_nonprintable(text_col, expected_text):
    df_test = pd.DataFrame({'text': text_col})
    res = rm_nonprintable(df_test.text)
    assert list(res.values) == expected_text  


@pytest.mark.parametrize("text_col, expected_text",
    [
        (['00-00-00', '12-A2-B50', '132.00-130.3444', '132.00-130,123,99+E50', '-132.00+34', '12(2lkj2)09'],
         ['00 - 00 - 00', '12-A2-B50', '132.00 - 130.3444', '132.00 - 130 , 123 , 99+E50', '-132.00 + 34', '12 ( 2lkj2 ) 09']
        )
    ]
)
def test_rm_mid_num_punct(text_col, expected_text):
    df_test = pd.DataFrame({'text': text_col})
    res = rm_mid_num_punct(df_test.text)
    assert list(res.values) == expected_text      


@pytest.mark.parametrize("text_col, expected_text",
    [
        # this function also removes double spaces
        (["cur-\nrent", "cur- \n rent", "cur; -\n rent"],
         ["current", "current", "cur; -\n rent"]
        )
    ]
)
def test_rm_newline_hyphenation(text_col, expected_text):
    df_test = pd.DataFrame({'text': text_col})
    res = rm_newline_hyphenation(df_test.text)
    assert list(res.values) == expected_text     


@pytest.mark.parametrize("text_col, expected_text",
    [
        (["curr-ent", "\"current\"", "cu\"()rr#en%tcur", "cur.ent,.", "cur,'ent.", "cur.,", "cur'"],
         ["curr-ent", "\"current\"", "currentcur", "cur.ent.", "cur'ent.", "cur.,", "cur'"]
        ),
        (["H23;0", "223+E02", "-23.0003"],
         ["H23;0", "223+E02", "-23.0003"]
        )
    ]
)
def test_rm_mid_word_punct(text_col, expected_text):
    df_test = pd.DataFrame({'text': text_col})
    res = rm_mid_word_punct(df_test.text)
    assert list(res.values) == expected_text   

@pytest.mark.parametrize("text_col, expected_text",
    [
        (["-+?curr-ent", "\"current\"", "curr-", "cur.ent.,", "cur,'", "cur.,", ".cur"],
         ["curr-ent", "\"current\"", "curr", "cur.ent.,", 'cur,', "cur.,", "cur"]
        ),
        (["H23;0", "223+E02", "-23.0003"],
         ["H23;0", "223+E02", "-23.0003"]
        )
    ]
)
def test_rm_beg_end_word_punct(text_col, expected_text):
    df_test = pd.DataFrame({'text': text_col})
    res = rm_beg_end_word_punct(df_test.text)
    assert list(res.values) == expected_text       


@pytest.mark.parametrize("text_col, expected_text",
    [
        (["-+?curr-ent", "\"current\"", "curr-", "cur.ent.,", "cur,'", "cur.,",],
         ["-?curr-ent", "current", "curr-", "cur.ent.,", 'cur,', "cur.,"]
        ),
        (["H23;0", "223+E02", "-23.0003"],
         ["H23;0", "223E02", "-23.0003"]
        )
    ]
)
def test_rm_non_textual_punct(text_col, expected_text):
    df_test = pd.DataFrame({'text': text_col})
    res = rm_non_textual_punct(df_test.text)
    assert list(res.values) == expected_text       


@pytest.mark.parametrize("text_col, expected_text",
    [
        (["-+?cu##r#r-ent", "\"cur?!rent\"", "curr-", "cur.ent.,", "cur,'", "cur.,",],
         ["curr-ent", "\"current\"", "curr", "cur.ent.,", 'cur,', "cur.,"]
        ),
        (["H23;0", "223+E02", "-23.0003"],
         ["H23;0", "223+E02", "-23.0003"]
        )
    ]
)
def test_rm_beg_end_word_punct_mid_word_punct(text_col, expected_text):
    df_test = pd.DataFrame({'text': text_col})
    res = rm_beg_end_word_punct(df_test.text)
    res = rm_mid_word_punct(res)
    print(res)
    assert list(res.values) == expected_text    

@pytest.mark.parametrize("text_col, expected_text",
    [
        (["""  withi n s h ort di stEm ce s and a t different dep t h s corr e s p onding""",
        """  withi n s h ort di stan ce s and a t different dep t h s corr e s p onding""",
        """  withi n s h ort di stEm ce s and a t different dep t h s corr e s p onding
            v P.riat ions in t he q_ua l i ty of t he ground wat e r derived f rom the
            drift Fir e to b e expec t ed One we l l mn.y y i el d A. moder r: t ely ha rd
            sli ghtly mine r e liz e d wRt e r whe r ea s P.nother we ll sunk to a similnr
            de::i th M.d lo c<- t ed only 50 f ee t dis t FLnt mny g iv e water th a t is too
            h i gh in dissolved su lph~ t e s s ~l ts to be used ei ther for drinking
            or stock w~ t e ring """, 
            """i n t h e y""",
            """  Wr1 ter the.t cont a i ns a l 'lrgo amoun t of s di w".l. carbo n-.l t e ::md
            sm~,1 1 '-tmounts of cr.:lcium -3.nd rrDgnesi um sr:i.l ts is sof t :mt if
            tht:, cal c i um 'md r.i'1gnesitm salt s a:r:, pr e s ent in l :"rge a.mo 11nt s
            t he wc.ter""",
            """and o nt he way""",
            """and a new day will be coming"""
            ],
         ["""within short di stEm ce sand at different depths corresponding""",
         """within short distances and at different depths corresponding""",
         """  within short di stEm ce sand at different depths corresponding
            v P.riat ions in the q_ua l i ty of the groundwater derived from the
            drift Fire to be expected One well mn.y yield A. moder r: t ely hard
            slightly miner e liz ed wRt er whereas P.nother well sunk to a similnr
            de::i th M.d lo c<- ted only 50 feet dist FLnt mny give water that is too
            high in dissolved su lph~ t es s ~l ts to be used either for drinking
            or stock w~ t e ring """, 
            """in they""",
            """  Wr1 ter the.t contains a l 'lrgo amount of s di w".l. carbo n-.l t e ::md
            sm~,1 1 '-tmounts of cr.:lcium -3.nd rrDgnesi um sr:i.l ts is soft :mt if
            tht:, calcium 'md r.i'1gnesitm salts a:r:, present in l :"rge a.mo 11nt st
            he wc.ter""",
            """and on the way""",
            """and anew day will be coming"""
            ]
        ),
    ]
)
def test_merge_words(text_col, expected_text):
    df_test = pd.DataFrame({'text': text_col})
    #df_test['text'] = df_test.text.str.replace('\s+', ' ')
    res = merge_words(df_test.text)

    expected_text = [re.sub(r'\s+', ' ', text).strip() for text in expected_text]
    assert list(res.values) == expected_text    


@pytest.mark.parametrize("text_col, expected_text",
    [
        (["""  withi n s h ort di stEm ce s and a t different dep t h s corr e s p onding""",
        """  withi n s h ort di stan ce s and a t different dep t h s corr e s p onding""",
        """  withi n s h ort di stEm ce s and a t different dep t h s corr e s p onding
            v P.riat ions in t he q_ua l i ty of t he ground wat e r derived f rom the
            drift Fir e to b e expec t ed One we l l mn.y y i el d A. moder r: t ely ha rd
            sli ghtly mine r e liz e d wRt e r whe r ea s P.nother we ll sunk to a similnr
            de::i th M.d lo c<- t ed only 50 f ee t dis t FLnt mny g iv e water th a t is too
            h i gh in dissolved su lph~ t e s s ~l ts to be used ei ther for drinking
            or stock w~ t e ring """, 
            """i n t h e y""",
            """  Wr1 ter the.t cont a i ns a l 'lrgo amoun t of s di w".l. carbo n-.l t e ::md
            sm~,1 1 '-tmounts of cr.:lcium -3.nd rrDgnesi um sr:i.l ts is sof t :mt if
            tht:, cal c i um 'md r.i'1gnesitm salt s a:r:, pr e s ent in l :"rge a.mo 11nt s
            t he wc.ter"""
            ],
         ["""within short di stEm ce sand at different depths corresponding""",
         """within short distance sand at different depths corresponding""",
         """  within short di stEm ce sand at different depths corresponding
            v P.riat ions in the q_ua l i ty of the groundwater derived from the
            drift Fire to be expected One well mn.y yield A. moder r: t ely hard
            slightly mine re liz ed wRt er whereas P.nother well sunk to a similnr
            de::i th M.d lo c<- ted only 50 feet dist FLnt mny give water that is too
            high in dissolved su lph~ t es s ~l ts to be used either for drinking
            or stock w~ t e ring """, 
            """in they""",
            """  Wr1 ter the.t contains a l 'lrgo amount of s di w".l. carbo n-.l t e ::md
            sm~,1 1 '-tmounts of cr.:lcium -3.nd rrDgnesi um sr:i.l ts is soft :mt if
            tht:, calcium 'md r.i'1gnesitm salts a:r:, present in l :"rge a.mo 11nt s
            the wc.ter"""
            ]
        ),
    ]
)
def test_merge_words_bkwd(text_col, expected_text):
    df_test = pd.DataFrame({'text': text_col})
    #df_test['text'] = df_test.text.str.replace('\s+', ' ')
    res = merge_words_bkwd(df_test.text)

    expected_text = [re.sub(r'[\s]+', ' ', text).strip() for text in expected_text]
    print(res)
    assert list(res.values) == expected_text  


@pytest.mark.parametrize("text_col, expected_text", [
    (["This a.j.hopkins@hopkin.hop.com name",
      "crazy@crazy-crazy.crazy-crazy.org",
      "This a.j.hopkins@hopkin.hop.com. This name",
      "This (crazy-@j.com) there"
    ],
    ["This name",
    " ",
    "This . This name",
    "This ( ) there"]),
])
def test_rm_email(text_col, expected_text):
    df_test = pd.DataFrame({'text': text_col})
    res = rm_email(df_test.text)
    assert list(res.values) == expected_text       

@pytest.mark.parametrize("text_col, expected_text", [
    (["This google.com name",
    "this www.gg-gg.com/com/com-g. name",
    " a-w.ca",
    "This (https://dd.d.org/10.0.0.0/303.0) there",
    "blah ftp://ftp2.cits.rncan.gc.ca/pub/geott/ess_pubs/214/214401/gscof_1746_e_2003_mn1.pdf blah",
    "ab out ground wo.ter in", # expect this one to be removed (it's a legal url, but a bad one)
    "co..lled ", # this should not be removed
    "O BARP.tt:GER R~S~ARCH",
    "1063028 490, rue de la Couronne Quebec (Quebec) G1K 9A9 Tel. : 418-654-2677 Telecopieur : 418-654-2660 Courriel : cgcq_librairie@rncan.gc.ca Web : http://www.cgcq.rncan.gc.ca/bibliotheque/"
    ],
    ["This name",
    "this . name",
    " ",
    "This ( ) there",
    "blah blah",
    "ab out ground in",
    "co..lled ",
    "O R~S~ARCH",
    "1063028 490, rue de la Couronne Quebec (Quebec) G1K 9A9 Tel. : 418-654-2677 Telecopieur : 418-654-2660 Courriel : Web : "
    ]),
])
def test_rm_url(text_col, expected_text):
    df_test = pd.DataFrame({'text': text_col})
    res = rm_url(df_test.text)
    assert list(res.values) == expected_text 

@pytest.mark.parametrize("text_col, expected_text", [
    (["This doi:10.1130/B30450.1 name",
    "this doi:10.1016/j.precamres.2009.04.005. name",
    " doi:10.1130/B30450.1",
    "This (doi:10.1130/B30450.1) there",
    "This (https://doi.org/10.4095/130284) there",
    "emediation, doi 10.1111/j.1745- 6592.1989.tb01125.x blah",
    "thidoi "
    ],
    ["This name",
    "this . name", #this one is sort of a problem
    " ",
    "This ( ) there",
    "This ( ) there",
    "emediation, - 6592.1989.tb01125.x blah", # this also happens to be covered by the doi removal
    "thidoi "
    ])
])
def test_rm_doi(text_col, expected_text):
    df_test = pd.DataFrame({'text': text_col})
    res = rm_doi(df_test.text)
    assert list(res.values) == expected_text 


@pytest.mark.parametrize("text_col, expected_text", [
    (["this 418-654-2660 is ",
      "this 1919-1920 is not a phonenumber",
      "this H202 is not",
      "this +798798095",
      " du Canada, 615, rue Booth, Ottawa, Ontario, K1A 0E9, telephone : (613) 995-5326, courriel ",
      "G1K 9A9 Tel. : 418-654-2677 Telecopieur :",
      "KIA 0E4 (613) 995-9351",
      ". Geotechnical profile for core 20130290075 fr",
      "gh natural diamonds 1983-1991 (million carats)",
      ". 32, p. 2057-2070.",
      "d 1988, GSC OF-1636, NGR-101-1988, NTS 131, 13J, rn::",
      "ments (anomalous) CAMS-17434 4040 +- 70",
      "on 5, p. 983-1006",
      " 63 km above mouth (1974-1992)"
    ],
    ["this is ",
      "this 1919-1920 is not a phonenumber",
      "this H202 is not",
      "this ",
      " du Canada, 615, rue Booth, Ottawa, Ontario, K1A 0E9, telephone : , courriel ",
      "G1K 9A9 Tel. : Telecopieur :",
      "KIA 0E4 ",
      ". Geotechnical profile for core 20130290075 fr",
      "gh natural diamonds 1983-1991 (million carats)",
      ". 32, p. 2057-2070.",
      "d 1988, GSC OF-1636, NGR-101-1988, NTS 131, 13J, rn::",
      "ments (anomalous) CAMS-17434 4040 +- 70",
      "on 5, p. 983-1006",
      " 63 km above mouth (1974-1992)"
    ])
])
def test_rm_phonenumber(text_col, expected_text):
    df_test = pd.DataFrame({'text': text_col})
    res = rm_phonenumber(df_test.text)
    assert list(res.values) == expected_text 

@pytest.mark.parametrize("text_col, expected_text", [
    (["this 418-654-2660 is ",
      "this 1919-1920 is not a phonenumber",
      "this/this is /"
      "ments (anomalous) CAMS-17434 4040 +- 70",
      "on 5, p. 983-1006",
      " 63 km above mouth (1974-1992)"
    ],
    ["this 418-654-2660 is ",
      "this 1919-1920 is not a phonenumber",
      "this this is "
      "ments (anomalous) CAMS-17434 4040 +- 70",
      "on 5, p. 983-1006",
      " 63 km above mouth (1974-1992)"
    ])
])
def test_rm_slash(text_col, expected_text):
    df_test = pd.DataFrame({'text': text_col})
    res = rm_slash(df_test.text)
    assert list(res.values) == expected_text 