# Copyright (C) 2021 ServiceNow, Inc.
""" Combine output datasets from different source datasets

    e.g. If you have generated training datasets for dataset A and dataset B
    you can combine them into A+B using this script

    It will *not* overwrite existing files (an error will be thrown). 
    Input files must exist (an error will be thrown otherwise). 
    It assumes that the output file will be saved to the same folder as the input
    (/nrcan_p2/data/03_primary/).
    It assumes nrcan specific file naming conventions.

    You MUST update the dataset parameters below.
"""
import pathlib
import subprocess 

################################### 
# DATASET PARAMETERS
PIPE = 'PIPELINE_BERT_80_POSTPIPE_BERT_SPACY_2' #'PIPELINE_GLOVE_80_POSTPIPE_GLOVE'
DATASET_A = 'dA_full_dB'
DATASET_B = 'dD'
DATASET_C = 'dA_full_dB_dD'
###################################

print('Combining files...')

DIR_MAPPING = {
    'dA_full': 'v4',
    'dB': 'v4_B',
    'dD': 'v4_D',
    'dA_full_dB': 'v4_A_B',
    'dA_full_dB_dD': 'v4_A_B_D'
}

DIR_A = DIR_MAPPING[DATASET_A] 
DIR_B = DIR_MAPPING[DATASET_B] 
DIR_C = DIR_MAPPING[DATASET_C] 


FILE_A = f'/nrcan_p2/data/03_primary/{DIR_A}/all_text_{PIPE}_{DATASET_A}_v1.txt'
print(FILE_A)

FILE_B = f'/nrcan_p2/data/03_primary/{DIR_B}/all_text_{PIPE}_{DATASET_B}_v1.txt'
print(FILE_B)

print('... into:')

FILE_C = f'/nrcan_p2/data/03_primary/{DIR_C}/all_text_{PIPE}_{DATASET_C}_v1.txt'
print(FILE_C)


file_a = pathlib.Path(FILE_A)
file_b = pathlib.Path(FILE_B)
file_c = pathlib.Path(FILE_C)

LOG_FILE = file_c.parent / (file_c.stem + '.log')

if not file_a.exists():
    raise(ValueError(f'File a does not exist: {FILE_A}'))

if not file_b.exists():
    raise(ValueError(f'File b does not exist: {FILE_B}'))


if file_c.exists():
    raise(ValueError(f'File c already exists. You must delete it manually: {FILE_C}'))

with open(LOG_FILE, 'w') as lf:
    lf.write(f'FILE_A: {FILE_A}\n')
    lf.write(f'FILE_B: {FILE_B}\n')
    lf.write(f'FILE_C: {FILE_C}\n')

with open(file_a, 'r') as fa, open(file_b, 'r') as fb, open(file_c, 'w') as fc:
    for line in fa:
        fc.write(line)

    for line in fb:
        fc.write(line)



if not file_c.exists():
    raise ValueError('ERROR! Something went wrong in the concatenation.')