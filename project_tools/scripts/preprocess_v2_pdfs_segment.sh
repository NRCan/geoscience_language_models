#!/bin/bash
# Copyright (C) 2021 ServiceNow, Inc.
#
# PDF->CSV
# 
UNPROCESSED=(
)

NOW=$(date +"%m_%d_%Y_%H_%M_%S")

make push

STEP=200
for start in $(seq 0 ${STEP} 3999)
do
    echo $start
    end=$(($start+${STEP}))
    echo $end

    #be *very* careful to put a / at the end of the filepath
    CMD="python scripts/pdf_to_txt.py --NFILE_RANGE $start $end --N_FILES -1 --WRITE_OUTPUT --LOCAL_DIR /nrcan_p2/data/01_raw/20201221/doaj/ --OUTPUT_DIR /nrcan_p2/data/02_intermediate/20201221/doaj/"

    echo "Error: Internal job management tool was to execute the command. Please run the command locally."
done
