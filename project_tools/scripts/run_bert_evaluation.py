# Copyright (C) 2021 ServiceNow, Inc.
""" Script for running bert keyword evaluation """

import pandas as pd

import nrcan_p2.evaluation.keyword_prediction as kp
from nrcan_p2.evaluation.keyword_prediction import run_keyword_prediction_bert

import argparse

def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


parser = argparse.ArgumentParser()
parser.add_argument('--TASK', help="evaluation task")
parser.add_argument('--DATA_DIR', help='data file')
parser.add_argument('--USE_CLASS_WEIGHTS', default=False, type=str2bool, help='use class weights')
parser.add_argument('--MODEL_PATH', help='model_path')

args = parser.parse_args()

METHOD = args.TASK
DATA_DIR = args.DATA_DIR 
USE_CLASS_WEIGHTS = args.USE_CLASS_WEIGHTS
MODEL_PATH = args.MODEL_PATH

print(METHOD)
print(DATA_DIR)
print(USE_CLASS_WEIGHTS)
print(MODEL_PATH)

if METHOD not in ['PAIRING', 'MULTICLASS']:
    raise ValueError(f'Unknown method {METHOD}')

# This key is no longer associated with any wandb account. You will need to add your own.


# Change below to use Weights & Biases for visualization
WANDB_API_KEY=None
WANDB_DIR=None  # path/to/save/the/output/files

# This data file must already have all null values dropped in the training set
if METHOD == 'PAIRING':

    eval_data, output_dir = run_keyword_prediction_bert(data_dir=DATA_DIR,
                                        output_dir='/nrcan_p2/data/07_model_output/keyword_prediction_bert/',
                                        keyword_text_col='keyword_text',
                                        keyword_cat_col='cat', 
                                        label_col='label',
                                        n_splits=5,
                                        n_rerun=3,
                                        task='PAIRING',
                                        WANDB_API_KEY=WANDB_API_KEY,
                                        WANDB_DIR=WANDB_DIR,
                                        model_path=MODEL_PATH,
                                        )

elif METHOD == 'MULTICLASS':

    eval_data, output_dir = run_keyword_prediction_bert(data_dir=DATA_DIR,
                                        output_dir='/nrcan_p2/data/07_model_output/keyword_prediction_bert/',
                                        keyword_text_col='keyword_text',
                                        keyword_cat_col='cat', 
                                        label_col='label',
                                        n_splits=5,
                                        n_rerun=3,
                                        task='MULTICLASS',
                                        WANDB_API_KEY=WANDB_API_KEY,
                                        WANDB_DIR=WANDB_DIR,
                                        use_class_weights=USE_CLASS_WEIGHTS,
                                        model_path=MODEL_PATH
                                        ) 


print(eval_data)
print(output_dir)
