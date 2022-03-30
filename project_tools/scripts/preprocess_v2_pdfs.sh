#!/bin/bash
# Copyright (C) 2021 ServiceNow, Inc.
#
# PDF->CSV 
# Convert all files in the specified folder
#
UNPROCESSED=(
)

CMD="python scripts/pdf_to_txt.py --N_FILES -1 --WRITE_OUTPUT --LOCAL_DIR /nrcan_p2/data/01_raw/20210108/ --OUTPUT_DIR /nrcan_p2/data/02_intermediate/20210108/"
echo "Error: Internal job management tool was to execute the command. Please run the command locally."
