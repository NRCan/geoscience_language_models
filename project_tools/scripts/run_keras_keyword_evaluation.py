# Copyright (C) 2021 ServiceNow, Inc.
""" Script to train a GloVe keyword prediction model using keras """ 
import nrcan_p2.evaluation.keyword_prediction as kp
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
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
parser.add_argument('--EMBEDDING_MODEL_PATH', help='model_path')
parser.add_argument('--EXISTING_RUN_DIR', default=None, help='existing run to finish')

args = parser.parse_args()

METHOD = args.TASK
DATA_DIR = args.DATA_DIR 
USE_CLASS_WEIGHTS = args.USE_CLASS_WEIGHTS
EMBEDDING_MODEL_PATH = args.EMBEDDING_MODEL_PATH
EXISTING_RUN_DIR = args.EXISTING_RUN_DIR

print(METHOD)
print(DATA_DIR)
print(USE_CLASS_WEIGHTS)
print(EMBEDDING_MODEL_PATH)
print(EXISTING_RUN_DIR)

output_dir='/nrcan_p2/data/07_model_output/keyword_prediction_keras/'

kp.run_keyword_prediction_keras(
    data_dir=DATA_DIR,
    output_dir=output_dir,
    n_splits=5,
    n_rerun=3,
    task=METHOD,
    use_class_weight=USE_CLASS_WEIGHTS,
    embedding_model_path=EMBEDDING_MODEL_PATH,
    njobs=None,
    existing_run_dir=EXISTING_RUN_DIR
)