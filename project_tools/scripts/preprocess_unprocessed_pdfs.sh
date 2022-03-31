#!/bin/bash
# Copyright (C) 2021 ServiceNow, Inc.
#
# PDF->CSV for specific file ids
# NOTE: this only words for files with numeric fileids

UNPROCESSED=(
    '224674' '247715' '287928' '288746' '288751' '288930' '288935' '289525' '289530' '292673'
)

CMD="python notebooks/pdf_to_txt.py --N_FILES -1 --WRITE_OUTPUT --LOCAL_DIR /nrcan_p2/data/01_raw/20201006/geoscan/raw/pdf --OUTPUT_DIR /nrcan_p2/data/02_intermediate/ --PDF_IDS ${UNPROCESSED[@]}"

echo "Error: Internal job management tool was to execute the command. Please run the command locally."
