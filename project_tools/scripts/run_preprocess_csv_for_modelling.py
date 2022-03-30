# Copyright (C) 2021 ServiceNow, Inc.
#
# Script for running preprocessing
# The less preferred alternative to run_preprocess_csv_for_modelling.sh
# 
# WARNING: run run_preprocess_csv_for_modelling.sh instead
#

# Whether or not to exclude the low text PDFS
# DO NOT set this to True
EXCLUDE_LOW = False

# The output folders
OUTPUT_DIR = '/nrcan_p2/data/03_primary/v4'
PARTIAL_OUTPUT_DIR = '/nrcan_p2/data/03_primary/v4'

# Pipeline settings
IS_PARTIAL = True
DELAY = 3600*3
i = 0
PERC_FILE_START = 75
PERC_FILE_END = 100
PERC_LOOP=5

# Input folders
INPUT_DIRS = [  '/nrcan_p2/data/02_intermediate/20201006/geoscan/pdf/v1_all', 
                '/nrcan_p2/data/02_intermediate/20201117/geoscan/pdf/generic_pdfs_all', 
                '/nrcan_p2/data/02_intermediate/20201117/geoscan/pdf/has_pdf_dir_all', 
                '/nrcan_p2/data/02_intermediate/20201117/geoscan/pdf/of_pdf_all']

# Pipelines
#PIPELINE = 'SIMPLE_PIPELINE_BERT_3'
PIPELINE = 'SIMPLE_PIPELINE_GLOVE_3'
#POST_PIPELINE = 'POSTPIPE_BERT_SPACY'
POST_PIPELINE = 'POSTPIPE_GLOVE'
SUFFIX = f'{PIPELINE}_{POST_PIPELINE}_dA_v1'

# Number of files 
N_FILES=-1 

########################

if not EXCLUDE_LOW: 
    INPUT_DIRS.append('/nrcan_p2/data/02_intermediate/20201117/geoscan/pdf/low_text_pdfs')
    

if EXCLUDE_LOW:
    SUFFIX = f"notlow_{SUFFIX}"


if IS_PARTIAL:
    INPUT_DIRS = [INPUT_DIRS[i]]
    SUFFIX = f"{SUFFIX}_partial"


from preprocess_csv_for_modelling import main

for perc_i in range(PERC_FILE_END,PERC_FILE_START,PERC_LOOP):
    print(perc_i)

import sys
sys.argv = ['preprocess_csv_for_modelling.py', '--INPUT_DIRS'] + INPUT_DIRS + [
    '--PARTIAL_OUTPUT_DIR', PARTIAL_OUTPUT_DIR,
    '--OUTPUT_DIR', OUTPUT_DIR, 
    '--PREPROCESSING_PIPELINE', PIPELINE,
    "--POST_PIPELINE", POST_PIPELINE,
    '--SUFFIX', SUFFIX,
    "--N_FILES", str(N_FILES),
    "--NO_FINAL_FILE", str(IS_PARTIAL),
    "--PERC_FILE_START", str(PERC_FILE_START),
    "--PERC_FILE_END", str(PERC_FILE_END)
]

print(sys.argv)

main()
