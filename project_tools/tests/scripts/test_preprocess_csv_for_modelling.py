# Copyright (C) 2021 ServiceNow, Inc.

import pytest
import pathlib

import pandas as pd
from nrcan_p2.data_processing.preprocessing_dfcol import rm_punct, lower
from scripts.preprocess_csv_for_modelling import build_next_temp_file_name, process_pre_pipeline_step

@pytest.mark.parametrize("input_file, orig_dir, new_output_dir, suffix, ext, expected",[
    (
    "/nrcan_p2/data/02_intermediate/106169_pa_79_05.csv",
    "/nrcan_p2/data/02_intermediate/",
    "/nrcan_p2/data/03_primary/v4",
    "0",
    ".csv",
    "/nrcan_p2/data/03_primary/v4/0/106169_pa_79_05.__0.csv"
    ),       
    (
    "/nrcan_p2/data/02_intermediate/0__1__2/106169_pa_79_05.__0__1__2.csv",
    "/nrcan_p2/data/02_intermediate/",
    "/nrcan_p2/data/03_primary/v4",
    "3",
    ".csv",
    "/nrcan_p2/data/03_primary/v4/0__1__2__3/106169_pa_79_05.__0__1__2__3.csv"
    ),           
    (
    "/nrcan_p2/data/02_intermediate/106169_pa_79_05.pdfminer_split.csv",
    "/nrcan_p2/data/02_intermediate/",
    "/nrcan_p2/data/03_primary/v4",
    "0",
    ".csv",
    "/nrcan_p2/data/03_primary/v4/0/106169_pa_79_05.pdfminer_split__0.csv"
    ),
    (
    "/nrcan_p2/data/02_intermediate/106169_pa_79_05__01.pdfminer_split.csv",
    "/nrcan_p2/data/02_intermediate/",
    "/nrcan_p2/data/03_primary/v4",
    "0",
    ".csv",
    "/nrcan_p2/data/03_primary/v4/0/106169_pa_79_05__01.pdfminer_split__0.csv"
    ),    
    ("/nrcan_p2/data/02_intermediate/20201006/geoscan/pdf/v1_all/277.test.csv",
     "/nrcan_p2/data/02_intermediate/",
     "/nrcan_p2/data/03_primary/v3/",
     "new_func",
     ".csv",
     "/nrcan_p2/data/03_primary/v3/new_func/20201006/geoscan/pdf/v1_all/277.test__new_func.csv"
    ),
    ("/nrcan_p2/data/02_intermediate/func_1/20201006/geoscan/pdf/v1_all/277.test__func_1.csv",
     "/nrcan_p2/data/02_intermediate/",
     "/nrcan_p2/data/03_primary/v3/",
     "func_2",
     ".csv",
     "/nrcan_p2/data/03_primary/v3/func_1__func_2/20201006/geoscan/pdf/v1_all/277.test__func_1__func_2.csv"
    ),
    ("/nrcan_p2/data/02_intermediate/func_1__func_2/20201006/geoscan/pdf/v1_all/277.test__func_1__func_2.csv",
     "/nrcan_p2/data/02_intermediate/",
     "/nrcan_p2/data/03_primary/v3/",
     "func_3",
     ".csv",
     "/nrcan_p2/data/03_primary/v3/func_1__func_2__func_3/20201006/geoscan/pdf/v1_all/277.test__func_1__func_2__func_3.csv"
    ),
    ("/nrcan_p2/data/02_intermediate/func_1__func_2/20201006/geoscan/pdf/v1_all/277_of__OF.test__func_1__func_2.csv",
     "/nrcan_p2/data/02_intermediate/",
     "/nrcan_p2/data/03_primary/v3/",
     "func_3",
     ".csv",
     "/nrcan_p2/data/03_primary/v3/func_1__func_2__func_3/20201006/geoscan/pdf/v1_all/277_of__OF.test__func_1__func_2__func_3.csv"
    )            
])
def test_build_next_temp_file_name(input_file, orig_dir, new_output_dir, suffix, ext, expected):
    input_file = pathlib.Path(input_file)
    orig_dir = pathlib.Path(orig_dir)
    new_output_dir = pathlib.Path(new_output_dir)

    result = build_next_temp_file_name(input_file, orig_dir, new_output_dir, suffix, ext)
    assert str(result) == expected


def test_process_pre_pipeline_step(tmp_path):
    input_file = tmp_path / "input_file.csv"
    col = "text"
    data = pd.DataFrame({col: ['Text1 .', 'Text2 .', 'text3 .']})
    data.to_csv(input_file)

    output_file = tmp_path / "output_file.csv"

    new_col = "processed_text"
    data[new_col] = ['Text1 ', 'Text2 ', 'text3 ']
    data = data.reset_index()
    data.columns = ['Unnamed: 0', 'text', 'processed_text']

    process_pre_pipeline_step(col, new_col, input_file, output_file, rm_punct)

    result_data = pd.read_csv(output_file)

    pd.testing.assert_frame_equal(result_data, data)

    # repeat with another function... 
    output_file2 = tmp_path / "output_file2.csv"

    col = "processed_text"
    new_col = "processed_text"

    process_pre_pipeline_step(col, new_col, output_file, output_file2, lower)

    data[new_col] = ['text1 ', 'text2 ', 'text3 ']

    result_data = pd.read_csv(output_file2)

    pd.testing.assert_frame_equal(result_data, data)    