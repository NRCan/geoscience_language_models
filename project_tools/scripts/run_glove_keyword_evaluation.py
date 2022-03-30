# Copyright (C) 2021 ServiceNow, Inc.
""" Script to train a GloVe keyword prediction model using sklearn """ 

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
parser.add_argument('--CLASSIFICATION_MODEL', help='Classification model (RF or MLP)')
parser.add_argument('--USE_MULTIOUTPUT_WRAPPER', help='use multioutput wrapper', type=str2bool, default=False)

args = parser.parse_args()

METHOD = args.TASK
DATA_DIR = args.DATA_DIR 
USE_CLASS_WEIGHTS = args.USE_CLASS_WEIGHTS
EMBEDDING_MODEL_PATH = args.EMBEDDING_MODEL_PATH
CLASSIFICATION_MODEL = args.CLASSIFICATION_MODEL
USE_MULTIOUTPUT_WRAPPER = args.USE_MULTIOUTPUT_WRAPPER

print(METHOD)
print(DATA_DIR)
print(USE_CLASS_WEIGHTS)
print(EMBEDDING_MODEL_PATH)
print(CLASSIFICATION_MODEL)
print(USE_MULTIOUTPUT_WRAPPER)


def rf_initializer(
    class_weight,
    njobs,
    random_state,
):
    params = {
        'max_depth': None,
        'n_estimators': 100,
    }
    gs_params = {
        'max_depth': [None, 1, 5, 10, 100],
        'n_estimators': [1, 5, 10, 50, 100]
    }
    return RandomForestClassifier(
        random_state=random_state, 
        max_depth=params['max_depth'], 
        n_estimators=params['n_estimators'], 
        class_weight=class_weight,
        verbose=True,
        n_jobs=njobs
    ), params, gs_params


def mlp_initializer(
    class_weight,
    njobs,
    random_state    
):
    params = dict(
        solver='adam',
        activation='relu',
        hidden_layer_sizes=(100,),
        max_iter=2000,
        early_stopping=True,   
        alpha=0.0001
    )
    gs_params = {
        'alpha': [0.0001, 0.001, 0.01],
        'max_iter': [200, 1000, 2000],
        'early_stopping': [False, True],
        'hidden_layer_sizes': [
            (100,),
            (50,),
            (100, 50),
            (300, 150),
        ]
    }
    return MLPClassifier(
        solver=params['solver'],
        activation=params['activation'],
        hidden_layer_sizes=params['hidden_layer_sizes'],
        #n_jobs=njobs,
        random_state=random_state,
        max_iter=params['max_iter'],
        early_stopping=params['early_stopping']
    ), params, gs_params

if CLASSIFICATION_MODEL == "RF":
    clf_initializer = rf_initializer
    output_dir='/nrcan_p2/data/07_model_output/keyword_prediction_glove/'
elif CLASSIFICATION_MODEL == "MLP":
    clf_initializer = mlp_initializer
    output_dir='/nrcan_p2/data/07_model_output/keyword_prediction_glove/'
else:
    raise ValueError(f"Unknown clf requested {CLASSIFICATION_MODEL}")

kp.run_keyword_prediction_classic(
    data_dir=DATA_DIR,
    output_dir=output_dir,
    n_splits=5,
    n_rerun=3,
    task=METHOD,
    use_class_weight=USE_CLASS_WEIGHTS,
    embedding_model_path=EMBEDDING_MODEL_PATH,
    clf_initializer=clf_initializer,
    vectorization_method='mean',
    njobs=None,
    use_multioutput_wrapper=USE_MULTIOUTPUT_WRAPPER
)