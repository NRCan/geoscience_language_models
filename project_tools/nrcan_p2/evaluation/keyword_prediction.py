# Copyright (C) 2021 ServiceNow, Inc.
""" Functionality for training all keyword prediction downstream models 
    and building the downstream task dataset
""" 
import pandas as pd
from typing import Union, List, Callable
import tqdm
import datetime
import random
import pathlib
import numpy as np
import subprocess
import sys
import joblib
import os
import re
import json
import wandb
import sklearn
import nrcan_p2.data_processing.pipeline_utilities as pu
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.model_selection import cross_validate 
from sklearn.model_selection import GroupShuffleSplit, GridSearchCV
from sklearn.model_selection import train_test_split
from filelock import FileLock
from imblearn.over_sampling import RandomOverSampler
from nrcan_p2.data_processing.vectorization import convert_dfcol_text_to_vector
from sklearn.metrics import (
    accuracy_score, 
    precision_recall_fscore_support,
    multilabel_confusion_matrix,
    confusion_matrix
)
from keras import backend as K
from tensorflow import keras
import tensorflow as tf
from tensorflow.keras import layers
import pandas as pd
import pathlib 

from gensim.test.utils import datapath, get_tmpfile
from gensim.models import KeyedVectors
from gensim.scripts.glove2word2vec import glove2word2vec

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.callbacks import Callback

from gensim.test.utils import datapath, get_tmpfile
from gensim.models import KeyedVectors
from gensim.scripts.glove2word2vec import glove2word2vec

#from sklearn.multioutput import MultiOutputClassifier
from nrcan_p2.evaluation.sklearn_multioutput import MultiOutputClassifier

from nrcan_p2.evaluation.bert_keyword_prediction import main as run_bert_multiclass_script

class SupressSettingWithCopyWarning:
    """ To be used in with blocks to suppress SettingWithCopyWarning """
    def __enter__(self):
        pd.options.mode.chained_assignment = None

    def __exit__(self, *args):
        pd.options.mode.chained_assignment = 'warn'


def produce_text_column(
    df:pd.DataFrame,
    text_column:str,
    text_column_is_list:bool,
    text_col_processing:str,  
    pre_pipeline:str,
    post_pipeline:str,      
):
    """ Given a raw metadata df, produce the "text" column,
        adding "keyword_text" column to the df.

        This assumes that the input text column is either a str 
        or a list of str that should be somehow converted to a single str

    :returns: input df, with an extra column 'keyword_text' with the 
        processed text from column  'text_column
    """
    output_col = 'keyword_text'

    # get the text
    df[output_col] = df[text_column]

    # the text column might be a list, 
    # convert to a string as necesssary, using the 
    # text_col_processing method 
    if text_column_is_list:
        if text_col_processing == 'join':
            df[output_col] = df[output_col].str.join(' ')
        elif text_col_processing == 'first':
            df[output_col] = df[output_col].str[0]
        else:
            raise ValueError('Unknown text_col_processing')

    if pre_pipeline is not None:
        dff = pu.run_pipeline(df[[output_col]],
            col=output_col,
            next_col=output_col,
            preprocessing_pipe=pre_pipeline)
    else:
        dff = df

    if post_pipeline is not None:
        dff[output_col] = dff.apply(
            lambda row: pu.run_pipeline(
                row.to_frame().transpose(), 
                col=output_col, 
                next_col=output_col,
                postmerge_preprocessing_pipe=post_pipeline),
            axis=1
        )

    df[output_col] = dff[output_col]

    return df

def produce_cat_column(
    df,
    keyword_col,
    pre_pipeline,
    post_pipeline
):
    """ Given a raw metadata df, produce the "category" column,
        adding "keyword_cat" column to the df.

        This assumes that the input category column is a list of
        strings.

    :returns: input df, with an extra column 'keyword_cat' with the 
        processed text from column indicated by 'keyword_col'
    """

    output_col = 'keyword_cat'

    df = df.copy()
    df[output_col] = df[keyword_col]

    if pre_pipeline is None and post_pipeline is None:
        return df

    # assume it's a list of keywords
    df_kw = df.explode(column=output_col)

    if pre_pipeline is not None:
        df_kw = pu.run_pipeline(df_kw[[output_col]],
            col=output_col,
            next_col=output_col,
            preprocessing_pipe=pre_pipeline)
       
    if post_pipeline is not None:
        df_kw[output_col] = df_kw.apply(
            lambda row: pu.run_pipeline(
                row.to_frame().transpose(), 
                col=output_col, 
                next_col=output_col,
                postmerge_preprocessing_pipe=post_pipeline),
            axis=1
        )

    df_kw = df_kw.reset_index().groupby(['index']).agg(lambda x: list(x))

    # the previous step inserts nan values into what should be empty lists. remove them
    df_kw[output_col] = df_kw[output_col].apply(lambda x: [xx for xx in x if xx is not None and type(xx) == 'str'])

    df[output_col] = df_kw[output_col]

    return df


def produce_keyword_classification_dataset_from_df(
    df_parquet:Union[str,pd.DataFrame],
    pre_pipeline:str,
    post_pipeline:str,
    cat_pre_pipeline:str, 
    cat_post_pipeline:str,
    text_column:str,
    text_column_is_list:bool,
    text_col_processing:str,
    keyword_col:str,
    n_categories:int,
    task:str,
    n_negative_sample:int,
    do_not_drop:bool=False,
):
    """ Produce a keyword classification dataset

    :param df_parquet: the raw metadata file for produce a keyword dataset
        as either a df or the name of a parquet file to load
    :param pre_pipeline: the name of an NRCan "pre" pipeline to be used 
        to process the text column. A pre pipeline is one that operates
        at the textbox level.
    :param post_pipeline: the name of an NRCan "post" pipeline to be used
        to process the text column after pre_pipeline. A post pipeline 
        is one that operates on the textboxes once merged, but will be 
        applied here per example in the input df
    :param cat_pre_pipeline: the name of an NRCan "pre" pipeline to be used
        to process the category column 
    :param cat_post_pipeline: the name of an NRCan "post" pipeline to be used
        to process the category column
    :param text_column: the name of the text column in the input
    :param text_column_is_list: whether or not the text column is a str 
        or a list of str
    :param keyword_col: the name of the keyword column in the input
    :param n_categories: the top-n categories to maintain
    :param task: the type of dataset to produce, MULTICLASS or PAIRING
    :param n_negative_samples: the number of negative samples for the PAIRING 
        task, None to get all negative samples
    :param do_not_drop: whether to not drop rows with null values

    :returns: df with the columns 
        MULTICLASS: 'keyword_text', 'cat_X' ... for each X in the output categories
            keyword_text is the text input
            cat_X is 0/1 indicating the presence of a category 
        PAIRING: 'keyword_text', 'cat', 'label' 
            keyword_text is the text input
            cat is the category name
            label is 0/1 to indicate whether the cat matches the keyword_text
    """
    if type(df_parquet) == str:
        df = pd.read_parquet(df_parquet)
    else:
        df = df_parquet

    with SupressSettingWithCopyWarning():
        df = produce_text_column(
                df,
                text_column=text_column,
                text_column_is_list=text_column_is_list,
                text_col_processing=text_col_processing,  
                pre_pipeline=pre_pipeline,
                post_pipeline=post_pipeline,
            )

    # get the subject
    # drop None values in the keywords
    with SupressSettingWithCopyWarning():
        if task == 'MULTICLASS':
            df['keyword_cat'] = df[keyword_col].apply(lambda x: [xx.strip() for xx in x if xx is not None] if x is not None else [])   
        else:
            df['keyword_cat'] = df[keyword_col].apply(lambda x: [xx.strip() for xx in x if xx is not None] if x is not None else x)

        df = produce_cat_column(
            df,
            keyword_col='keyword_cat',
            pre_pipeline=cat_pre_pipeline,
            post_pipeline=cat_post_pipeline,        
        )

    vc = df['keyword_cat'].explode().value_counts()
    if n_categories is None:
        vc_subset = vc.index
    else:
        vc_subset = vc.index[0:n_categories]

    if task == "MULTICLASS":
        assert df.index.is_unique
        mlb = MultiLabelBinarizer()

        # multiclass classifier, produce one column per label
        if not do_not_drop:
            print(df.shape)
            df = df.dropna(subset=['keyword_cat'])
            print(df.shape)

        t = mlb.fit_transform(df.keyword_cat)
        
        Y = pd.DataFrame(t,columns=['cat_' + c for c in mlb.classes_])
        Y = Y[['cat_' + c for c in vc_subset]]
        df_ret = pd.merge(df, Y, right_index=True, left_index=True)

    elif task == "PAIRING":
        if not do_not_drop:
            print('Dropping...')
            print(df.shape)
            df = df.dropna(subset=['keyword_cat'])
            print(df.shape)

        full_vc_set = set(vc_subset)

        if n_negative_sample is not None:
            def get_sampled_categories(x):
                # Sample the desired number of negative examples
                rest = full_vc_set.difference(x)

                # if there are more elements than the sample we want, take a sample
                if len(rest) > n_negative_sample:
                    rest = random.sample(full_vc_set.difference(x),n_negative_sample) 
                
                # otherwise, just use the full set
                # probably unnecessary to check for nulls, but just in case...
                if len(rest) == 0:
                    return None
                else:
                    return rest

            df['cat_negative_sample'] = df.keyword_cat.apply(get_sampled_categories)
        
        else:
            def get_remaining_categories(x):
                # Produce all negative examples
                rest = list(full_vc_set.difference(x))
                if len(rest) == 0:
                    return None
                else:
                    return rest

            df['cat_negative_sample'] = df.keyword_cat.apply(get_remaining_categories)
            print('Dropping negative samples...')
            print(df.shape)
            df = df.dropna(subset=['cat_negative_sample'])
            print(df.shape)
        
        df_pos = df.explode(column='keyword_cat')
        df_pos['label'] = 1
        df_pos['cat'] = df_pos['keyword_cat']

        df_neg = df.explode(column='cat_negative_sample')
        df_neg['label'] = 0
        df_neg['cat'] = df_neg['cat_negative_sample']

        df_ret = pd.concat([df_pos, df_neg])
        df_ret = df_ret.drop(columns=['cat_negative_sample', 'keyword_cat']) #'cat_negative', 

    elif task == "PREDICT":
        raise NotImplementedError()

    return df_ret

def load_glove_model(model_path):
    print(f'Loading model from {model_path}...')
    glove_file = datapath(model_path)

    tmp_file_name = f"{pathlib.Path(model_path).parent}/tmp_word2vec.txt"
    with FileLock(str(tmp_file_name) + ".lock"):
        tmp_file = get_tmpfile(tmp_file_name)

        _ = glove2word2vec(glove_file, tmp_file)

        model = KeyedVectors.load_word2vec_format(tmp_file) 

    return model


class KerasValidMetrics(Callback):

    def __init__(self, val_data, batch_size = 32):
        super().__init__()
        self.validation_data = val_data
        #self.batch_size = batch_size

    def on_train_begin(self, logs={}):
        self.val_micro_precision = []
        self.val_micro_recall = []
        self.val_micro_fb1 = []

        self.val_macro_precision = []
        self.val_macro_recall = []
        self.val_macro_fb1 = []

        self.val_sample_precision = []
        self.val_sample_recall = []
        self.val_sample_fb1 = []        
        self.val_sample_support = []

        self.val_accuracy = []
    
    def on_epoch_end(self, epoch, logs={}):
        val_predict = (np.asarray(self.model.predict(self.validation_data[0]))).round()
        val_targ = self.validation_data[1]

        metrics = compute_metrics_multiclass(val_targ, val_predict)

        self.val_micro_precision.append(metrics['micro-precision'])
        self.val_macro_precision.append(metrics['macro-precision'])

        self.val_micro_recall.append(metrics['micro-recall'])
        self.val_macro_recall.append(metrics['macro-recall'])

        self.val_micro_fb1.append(metrics['micro-fb1'])
        self.val_macro_fb1.append(metrics['macro-fb1'])

        self.val_accuracy.append(metrics['accuracy'])

        self.val_sample_precision.append(metrics['sample-precision'])
        self.val_sample_recall.append(metrics['sample-recall'])
        self.val_sample_fb1.append(metrics['sample-fb1'])
        self.val_sample_support.append(metrics['support'])

        print(f" - val_micro-precision: {metrics['micro-precision']} - val_micro-recall: {metrics['micro-recall']} - val_micro_fb1: {metrics['micro-fb1']}")
        print(f" - val_macro-precision: {metrics['macro-precision']} - val_macro-recall: {metrics['macro-recall']} - val_macro_fb1: {metrics['macro-fb1']}")
        print(f" - val_accuracy: {metrics['accuracy']}")
        print(f" - val_sample_precision: {metrics['sample-precision']}")
        print(f" - val_sample_recall: {metrics['sample-recall']}")
        print(f" - val_sample_fb1: {metrics['sample-fb1']}")
        print(f" - val_sample_support: {metrics['support']}")
        return
 

def run_keyword_prediction_keras(
    data_dir,
    output_dir,
    n_splits,
    n_rerun=5,
    keyword_text_col='sentence1',
    label_col='label',
    keyword_cat_col='cat',
    task='MULTICLASS',
    use_class_weight=False,
    embedding_model_path=None,
    njobs=None,
    existing_run_dir=None,
):
    """ Train a model with keras """
    saved_args = locals()

    maxlen = 200

    if existing_run_dir is not None:
        print('Starting from an existing run...')
        output_dir = pathlib.Path(existing_run_dir)
        assert output_dir.exists()
    else:
        now = datetime.datetime.today().strftime('%Y-%m-%d-%H-%M-%S%f')
        output_dir_parent = pathlib.Path(output_dir)
        output_dir = pathlib.Path(output_dir) / f'run-glove-keras-{task}_{now}'
        output_dir.mkdir(parents=False, exist_ok=False)

    input_df_log = output_dir / "input_data.log"
    if not input_df_log.exists():
        with open(input_df_log, 'w') as f:
            json.dump({k: v.__name__ if callable(v) else v
                for k,v in saved_args.items()}, f, indent=4)

        embedding_model = load_glove_model(embedding_model_path)

    else:
        with open(input_df_log) as f:
            loaded_args = json.load(f)

        data_dir = loaded_args['data_dir']
        n_splits = loaded_args['n_splits']
        n_rerun = loaded_args['n_rerun']
        keyword_text_col = loaded_args['keyword_text_col']
        label_col = loaded_args['label_col']
        keyword_cat_col = loaded_args['keyword_cat_col']
        task = loaded_args['task']
        use_class_weight = loaded_args['use_class_weight']
        assert type(use_class_weight) == bool
        embedding_model_path = loaded_args['embedding_model_path']
        njobs = loaded_args['njobs']
        print('replacing...')
        print(saved_args)
        print('with..')
        print(loaded_args)

        embedding_model = load_glove_model(embedding_model_path)
            

    data_dir = pathlib.Path(data_dir)

    models_all = {}
    cv_scores_all = {}
    for i in range(0,n_splits):
        suffix = '.csv'

        print('--------------------------------------------------')
        print(f"Training split {i}...")
        train_file = data_dir / f"split_{i}" / ("train" + suffix)
        print(f"Train file: {train_file}")
        train_df = pd.read_csv(train_file)

        valid_file = data_dir / f"split_{i}" / ("valid" + suffix)
        print(f"Valid file: {valid_file}")
        valid_df = pd.read_csv(valid_file)
        valid_df = valid_df.fillna("")

        # we let the tokenizer build on both the train/val because we might 
        # have representations for tokens in val in our embedding already
        # but these might not be in the training set 
        print('Building tokenizer...')
        tokenizer = Tokenizer(
            num_words=None,
            filters="",
            lower=True,
            split=" ",
            char_level=False,
            oov_token=None,
            document_count=0,
        )
        tokenizer.fit_on_texts(pd.concat([train_df,valid_df])[keyword_text_col].values)

        train_sequences_sent1 = tokenizer.texts_to_sequences(train_df[keyword_text_col].values)
        valid_sequences_sent1 = tokenizer.texts_to_sequences(valid_df[keyword_text_col].values)

        word_index = tokenizer.word_index
        print(f'Found {len(word_index)} unique tokens in {keyword_text_col}.')

        if task == 'MULTICLASS':
            X_train = keras.preprocessing.sequence.pad_sequences(train_sequences_sent1, maxlen=maxlen)
            X_test = keras.preprocessing.sequence.pad_sequences(valid_sequences_sent1, maxlen=maxlen)

            print(X_train.shape)
            print(X_test.shape)

            cols = train_df.filter(regex=keyword_cat_col).columns
            
            Y_train = train_df.loc[:,cols].values
            Y_test = valid_df.loc[:,cols].values     

            class_weights = Y_train.shape[0]/Y_train.sum(axis=0) 
            class_weights_per_datum = np.dot(Y_train, class_weights)        
        elif task == 'PAIRING': 
            raise NotImplementedError()                            
        else:
            raise ValueError(f'Unknown task {task}')            

        models={}
        cv_scores={}
        for j in range(n_rerun):

            print(f"Training rerun {j}...")
            sub_output_dir = output_dir / f"split_{i}_run_{j}"   
            print(f'...{sub_output_dir}')

            if existing_run_dir is not None:
                model_save_name = pathlib.Path(sub_output_dir) / "model.keras"
                model_cv_results_file = pathlib.Path(sub_output_dir) / "model_history.json"
                scores_file = pathlib.Path(sub_output_dir) / "metrics.json"
                if model_save_name.exists() and model_cv_results_file.exists() and scores_file.exists():
                    print(f'...already trained for split {i} run {j}. Skipping...')
                    continue
            else:
                sub_output_dir.mkdir(parents=False, exist_ok=False)                    

            keras_model = build_keras_model(
                embedding_model=embedding_model,
                task=task,
                output_classes_shape=Y_train.shape[1]
            ) 
           
            def weighted_binary_crossentropy(y_true, y_pred, class_weights):
                return K.mean(K.binary_crossentropy(tf.cast(y_true, tf.float32), y_pred) * class_weights, axis=-1)

            if use_class_weight: 
                print(f'Training using class weighted loss..')
                loss = lambda y_true, y_pred: weighted_binary_crossentropy(y_true, y_pred, class_weights)
            else:
                print(f'Training using normal un-class weighted loss..')
                loss = 'binary_crossentropy'

            keras_model.compile("adam", loss=loss)                    

            model, cv_score = run_keras_model_cv(
                X_train=X_train, Y_train=Y_train, X_test=X_test, Y_test=Y_test,
                output_dir=sub_output_dir,
                model=keras_model,
                random_state=j,
                njobs=njobs,
            )

            models[j] = model
            cv_scores[j] = cv_score
        cv_scores_all[i] = cv_scores
        models_all[i] = models

    return models_all, cv_scores_all    

def run_keyword_prediction_classic(
    data_dir,
    output_dir,
    clf_initializer,
    n_splits,
    n_rerun=5,
    keyword_text_col='sentence1',
    label_col='label',
    keyword_cat_col='cat',
    task='MULTICLASS',
    use_class_weight=False,
    embedding_model_path=None,
    njobs=None,
    use_multioutput_wrapper=False,
    vectorization_method='sum',
):
    """ Train a model using sklearn """
    saved_args = locals()

    now = datetime.datetime.today().strftime('%Y-%m-%d-%H-%M-%S%f')
    output_dir_parent = pathlib.Path(output_dir)
    output_dir = pathlib.Path(output_dir) / f'run-glove-{task}_{now}'
    output_dir.mkdir(parents=False, exist_ok=False)

    embedding_model = load_glove_model(embedding_model_path)

    input_df_log = output_dir / "input_data.log"
    with open(input_df_log, 'w') as f:
        json.dump({k: v.__name__ if callable(v) else v
            for k,v in saved_args.items()}, f, indent=4)

    data_dir = pathlib.Path(data_dir)

    models_all = {}
    cv_scores_all = {}
    for i in range(0,n_splits):
        suffix = '.csv'

        print('--------------------------------------------------')
        print(f"Training split {i}...")
        train_file = data_dir / f"split_{i}" / ("train" + suffix)
        print(f"Train file: {train_file}")
        train_df = pd.read_csv(train_file)

        valid_file = data_dir / f"split_{i}" / ("valid" + suffix)
        print(f"Valid file: {valid_file}")
        valid_df = pd.read_csv(valid_file)

        if task == 'MULTICLASS':
            X_train = convert_dfcol_text_to_vector(train_df, keyword_text_col, embedding_model, method=vectorization_method)

            # fillna if necessary
            valid_df[keyword_text_col] = valid_df[keyword_text_col].fillna('')
            X_test = convert_dfcol_text_to_vector(valid_df, keyword_text_col, embedding_model, method=vectorization_method)                

            cols = train_df.filter(regex=keyword_cat_col).columns
            
            Y_train = train_df.loc[:,cols].values
            Y_test = valid_df.loc[:,cols].values     

            class_weights = Y_train.shape[0]/Y_train.sum(axis=0) 
            class_weights_per_datum = np.dot(Y_train, class_weights)

        elif task == 'PAIRING':
            X1 = convert_dfcol_text_to_vector(train_df, keyword_text_col, embedding_model, method=vectorization_method)
            X2 = convert_dfcol_text_to_vector(train_df, cat_text_col, embedding_model, method=vectorization_method)

            X_train= np.concatenate([X1, X2], axis=1)
            Y_train = train_df[label_col].values     

            X1 = convert_dfcol_text_to_vector(valid_df, keyword_text_col, embedding_model, method=vectorization_method)
            X2 = convert_dfcol_text_to_vector(valid_df, cat_text_col, embedding_model, method=vectorization_method)

            X_test= np.concatenate([X1, X2], axis=1)
            Y_test = valid_df[label_col].values                                   
        else:
            raise ValueError(f'Unknown task {task}')        

        models={}
        cv_scores={}
        for j in range(n_rerun):

            print(f"Training rerun {j}...")
            sub_output_dir = output_dir / f"split_{i}_run_{j}"   
            print(f'...{sub_output_dir}')
            sub_output_dir.mkdir(parents=False, exist_ok=False)                    

            model, cv_score = run_sklearn_model_cv(
                X_train=X_train, Y_train=Y_train, X_test=X_test, Y_test=Y_test,
                output_dir=sub_output_dir,
                use_class_weight=use_class_weight,
                class_weights_per_datum=class_weights_per_datum,
                clf_initializer=clf_initializer,
                random_state=j,
                njobs=njobs,
                use_multioutput_wrapper=use_multioutput_wrapper
            )

            models[j] = model
            cv_scores[j] = cv_score
        cv_scores_all[i] = cv_scores
        models_all[i] = models

    return models_all, cv_scores_all

def compute_metrics_multiclass(y_true, y_pred):
    """ Compute multiclass metrics """
    micro_precision, micro_recall, micro_fb1, support = precision_recall_fscore_support(y_true, y_pred, average='micro')
    macro_precision, macro_recall, macro_fb1, support = precision_recall_fscore_support(y_true, y_pred, average='macro')
    weighted_precision, weighted_recall, weighted_fb1, support = precision_recall_fscore_support(y_true, y_pred, average='weighted')
    sample_precision, sample_recall, sample_fb1, support = precision_recall_fscore_support(y_true, y_pred, average=None)
    confusion_matrix = multilabel_confusion_matrix(y_true, y_pred)

    return {'accuracy': accuracy_score(y_true, y_pred),
        'micro-precision': micro_precision,
        'micro-recall': micro_recall,
        'micro-fb1': micro_fb1,
        'macro-precision': macro_precision,
        'macro-recall': macro_recall,
        'macro-fb1': macro_fb1,
        'support': support.tolist(),
        'sample-precision': sample_precision.tolist(),
        'sample-recall': sample_recall.tolist(), 
        'sample-fb1': sample_fb1.tolist(),
        'confusion_matrix': confusion_matrix.tolist()
    }    

class NumpyEncoder(json.JSONEncoder):
    """ For saving dict with numpy arrays to json """
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


def build_keras_model(
    embedding_model,
    task,
    output_classes_shape,
):
    """ Build the keras model """
    if task == 'MULTICLASS':
        inputs = keras.Input(shape=(None,), dtype="int32")
        embedding_layer = embedding_model.get_keras_embedding()    
        x = embedding_layer(inputs)
        x = layers.Bidirectional(layers.LSTM(64))(x)
        outputs = layers.Dense(output_classes_shape, activation="sigmoid")(x)
        model = keras.Model(inputs, outputs)
        model.summary()        

    elif task == 'PAIRING':
        inputs = keras.Input(shape=(None,), dtype="int32")
        embedding_layer = embedding_model.get_keras_embedding()    
        x = embedding_layer(inputs)
        x = layers.Bidirectional(layers.LSTM(64))(x)

        y = embedding_layer(inputs2)
        y = layers.Bidirectional(layers.LSTM(64))(x)

        x = layers.Concatenate(axis=-1)([x,y])

        outputs = layers.Dense(y_train.shape[1], activation="sigmoid")(x)
        model = keras.Model(inputs, outputs)
        model.summary()             

    return model

def run_keras_model_cv(
    X_train, Y_train, X_test, Y_test,
    output_dir,
    model,
    random_state:float,
    njobs=None,
):
    """ Run cross validation for a single keras model """
    print(X_train.shape, Y_train.shape, X_test.shape, Y_test.shape)

    X_train_train, X_train_valid, Y_train_train, Y_train_valid = train_test_split(X_train, 
        Y_train, test_size=0.1, random_state=random_state)

    keras_valid_metrics = KerasValidMetrics(val_data=(X_train_valid, Y_train_valid))

    history = model.fit(X_train_train, Y_train_train, batch_size=32, epochs=200, 
            validation_data=(X_train_valid, Y_train_valid),
            callbacks=[keras_valid_metrics])

    # save the model
    model_save_name = pathlib.Path(output_dir) / "model.keras"
    print(f"Saving model to {model_save_name}")
    model.save(model_save_name)

    model_cv_results_file = pathlib.Path(output_dir) / "model_history.json"  
    with open(model_cv_results_file, 'w') as f:
        json.dump(history.history, f, indent=4, cls=NumpyEncoder)

    preds_prob = model.predict(X_test)
    preds = (preds_prob > 0.5).astype(int)

    preds_prob_x = model.predict(X_train)
    preds_x = (preds_prob_x > 0.5).astype(int)
    print(preds_x.shape, preds.shape)

    score_values = {}
    test_metrics = compute_metrics_multiclass(Y_test, preds)
    for metric_name,metric_value in test_metrics.items():
        metric_name = f'eval_{metric_name}'
        score_values[metric_name] = metric_value
        print(f'{metric_name}: {score_values[metric_name]}')

    train_metrics = compute_metrics_multiclass(Y_train,preds_x)
    for metric_name,metric_value in train_metrics.items():
        metric_name = f'train_{metric_name}'
        score_values[metric_name] = metric_value
        print(f'{metric_name}: {score_values[metric_name]}')            

    scores_file = pathlib.Path(output_dir) / "metrics.json"
    print(f"Writing metrics to {scores_file}...")
    with open(scores_file, 'w') as f:
        json.dump(score_values, f, indent=4)

    return model, score_values


def run_sklearn_model_cv(
    X_train, Y_train, X_test, Y_test,
    output_dir,
    clf_initializer,
    use_class_weight:bool,
    class_weights_per_datum:np.ndarray, #only used if use_class_weight is true and is necessary
    random_state:float,
    njobs=None,
    use_multioutput_wrapper=False,
):
    """ Run cross validation for a single sklearn model """
    print(X_train.shape, Y_train.shape, X_test.shape, Y_test.shape)

    clf, params, gs_params = clf_initializer(class_weight="balanced" if use_class_weight else None,
                        njobs=njobs, 
                        random_state=random_state)

    using_mlp = type(clf) in [sklearn.neural_network.MLPClassifier]

    if use_multioutput_wrapper:
        print('Wrapping the classifier with a multioutput classifier...')
        gs_params = {f'estimator__{key}':value for key,value in gs_params.items()}
        clf = MultiOutputClassifier(clf, n_jobs=njobs)

    gs = GridSearchCV(clf, gs_params, refit='f1_macro', cv=5, 
        return_train_score=True, 
        scoring=['accuracy', 'f1_macro', 'f1_micro', 'f1_samples', 
                'recall_macro', 'recall_micro', 'recall_samples',
                'precision_macro', 'precision_micro', 'precision_samples']) 

    if use_class_weight and using_mlp:
        print('Training with sample weights....')
        ros = RandomOverSampler(random_state=random_state)     
        if use_multioutput_wrapper:
            print('..Training with multioutput oversampling...')
            gs.fit(X_train,Y_train, imbalanced_sampler=ros)

        else:
            print('..NN dont support sampling')
            gs.fit(X_train, Y_train)

    else:
        print('Training without sample_weights...')
        gs.fit(X_train,Y_train)

    # save the model
    model_save_name = pathlib.Path(output_dir) / "model.joblib"
    print(f"Saving model to {model_save_name}")
    joblib.dump(clf, model_save_name)

    gs_save_name = pathlib.Path(output_dir) / "gs.joblib"
    print(f"Saving gs to {gs_save_name}")
    joblib.dump(gs, gs_save_name)    

    model_cv_results_file = pathlib.Path(output_dir) / "model_cv_results.json"
    with open(model_cv_results_file, 'w') as f:
        json.dump(gs.cv_results_, f, indent=4, cls=NumpyEncoder)

    model_params = pathlib.Path(output_dir) / "model_params.json"
    with open(model_params, 'w') as f:
        json.dump(gs.best_params_, f, indent=4, cls=NumpyEncoder)

    preds = gs.predict(X_test)
    preds_x = gs.predict(X_train)
    print(preds_x.shape, preds.shape)

    score_values = {}
    test_metrics = compute_metrics_multiclass(Y_test, preds)
    for metric_name,metric_value in test_metrics.items():
        metric_name = f'eval_{metric_name}'
        score_values[metric_name] = metric_value
        print(f'{metric_name}: {score_values[metric_name]}')

    train_metrics = compute_metrics_multiclass(Y_train,preds_x)
    for metric_name,metric_value in train_metrics.items():
        metric_name = f'train_{metric_name}'
        score_values[metric_name] = metric_value
        print(f'{metric_name}: {score_values[metric_name]}')            

    scores_file = pathlib.Path(output_dir) / "metrics.json"
    print(f"Writing metrics to {scores_file}...")
    with open(scores_file, 'w') as f:
        json.dump(score_values, f, indent=4)

    return clf, score_values


def produce_dataset_splits(
    output_dir:str, # the output dir for the data
    input_df_file:str, # the input file
    output_name:str, # the name of the output subfolder
    task:str, # PAIRING or MULTICLASS,
    keyword_text_col='keyword_text',
    label_col='label',
    keyword_cat_col='cat',
    n_splits=5,
    test_size=0.8,
    dropna=False,
    model_type="BERT"
):
    output_dir = pathlib.Path(output_dir) / output_name
    if output_dir.exists():
        raise ValueError(f'ERROR: output dir {output_dir} already exists')
    if not output_dir.exists():
        output_dir.mkdir(parents=False, exist_ok=False)

    input_df_log = output_dir / "input_data.log"
    with open(input_df_log, 'w') as f:
        f.write(str(input_df_file))

    df = pd.read_parquet(input_df_file)
    print(df.shape)

    if task == 'PAIRING':
        df = df[[label_col, keyword_text_col, keyword_cat_col]]
        df = df.rename(columns={label_col: 'label', keyword_text_col: 'sentence1', keyword_cat_col:'sentence2'})
        if dropna:
            df = df.dropna()
        else:
            df['sentence1'] = df.sentence1.fillna('')
            df['sentence2'] = df.sentence2.fillna('')
        X = df.index
    elif task == 'MULTICLASS':
        if model_type == "GLOVE":
            df = df.filter(regex=f'{keyword_cat_col}|{keyword_text_col}')
            df = df.rename(columns={keyword_text_col: 'sentence1'})
            if dropna:
                df = df.dropna()
        else:
            df = df[[label_col, keyword_text_col]]
            df = df.rename(columns={label_col: 'label', keyword_text_col: 'sentence1'})
            if dropna:
                df = df.dropna()
        X = df.index
    else:
        raise ValueError(f'Unknown task {task}')

    print(df.shape)

    if task == 'PAIRING': 
        gss = GroupShuffleSplit(n_splits=n_splits, test_size=test_size, random_state=0)
    elif task == 'MULTICLASS':
        gss = KFold(n_splits=n_splits, random_state=0)


    for i, (train_idx, test_idx) in enumerate(gss.split(X, groups=list(df.index))):        

            print(f'Split {i}')
            data_dir = output_dir / f"split_{i}"
            data_dir.mkdir(parents=False, exist_ok=False)

            df_split_train = df.iloc[train_idx,:]
            df_split_test = df.iloc[test_idx,:]

            # Always dropnas from the training set (but not the validation set, which is always the same)
            print('Dropping nas from the training set...')
            print(df_split_train.shape)
            df_split_train = df_split_train.dropna()
            print(df_split_train.shape)

            if task == 'PAIRING' or model_type == "GLOVE":
                train_file = data_dir / "train.csv"
                df_split_train.to_csv(train_file, index=False)
                print(f'Writing {train_file}...')
                print(df_split_train.shape)

                valid_file = data_dir / "valid.csv"
                df_split_test.to_csv(valid_file, index=False)
                print(f'Writing {valid_file}...')
                print(df_split_test.shape)

            elif task == 'MULTICLASS':
                train_file = data_dir / "train.json"
                df_split_train.to_json(train_file, index=False, orient='table', indent=4)
                print(f'Writing {train_file}...')
                print(df_split_train.shape)

                valid_file = data_dir / "valid.json"
                df_split_test.to_json(valid_file, index=False, orient='table', indent=4)
                print(f'Writing {valid_file}...')
                print(df_split_test.shape)   

     

def run_keyword_prediction_bert(
    data_dir,
    output_dir,
    n_splits=5,
    n_rerun=5,
    keyword_text_col='keyword_text',
    label_col='label',
    keyword_cat_col='cat',
    task='PAIRING',
    WANDB_API_KEY=None,
    WANDB_DIR=None,
    use_class_weights=False,
    model_path=None
):
    """ Finetune a bert model for keyword prediction """
    if model_path is None:
        raise ValueError(f'Must provide a model_path: {model_path}')

    now = datetime.datetime.today().strftime('%Y-%m-%d-%H-%M-%S%f')
    output_dir_parent = pathlib.Path(output_dir)
    output_dir = pathlib.Path(output_dir) / f'run-{task}_{now}'
    output_dir.mkdir(parents=False, exist_ok=False)

    if WANDB_API_KEY is not None:
        os.environ['WANDB_API_KEY'] = WANDB_API_KEY  
        os.environ['WANDB_DIR'] = WANDB_DIR

    input_df_log = output_dir / "input_data.log"
    with open(input_df_log, 'w') as f:
        f.write(data_dir)

    data_dir = pathlib.Path(data_dir)

    for i in range(0,n_splits):
        if task == 'PAIRING':
            suffix = '.csv'
        elif task == 'MULTICLASS':
            suffix = '.json'
            
        train_file = data_dir / f"split_{i}" / ("train" + suffix)
        print(f"Train file: {train_file}")

        valid_file = data_dir / f"split_{i}" / ("valid" + suffix)
        print(f"Valid file: {valid_file}")

        for j in range(n_rerun):     

            sub_output_dir = output_dir / f"split_{i}_run_{j}"   
            sub_output_dir.mkdir(parents=False, exist_ok=False)

            curr_folder = pathlib.Path(__file__).resolve().parent
            print('Training...')
            argv = [
                str(curr_folder / 'bert_keyword_prediction.py'),
                '--train_file', str(train_file),
                '--validation_file', str(valid_file),
                '--test_file',  str(valid_file),
                '--label_column_name', 'label',
                '--model_name_or_path', model_path,
                '--output_dir', str(sub_output_dir / "model"),
                '--per_device_train_batch_size', '32' ,
                '--per_device_eval_batch_size', '64' ,
                '--do_train' ,
                '--do_predict' ,
                "--seed", str(j),
                '--do_eval',
                '--logging_dir', str(sub_output_dir/ "logging"),
                '--fp16', #'True',
                '--logging_steps', '50' ,
                '--learning_rate', '2e-5' ,
                '--num_train_epochs', '4' ,
                '--cache_dir', str(output_dir_parent / "cache"),
                '--dataset_cache_dir', str(output_dir_parent / "datasets"),
                '--save_steps', '100' ,
                '--evaluation_strategy', 'steps',
                #'--overwrite_output_dir',
                '--max_seq_length', '512',
                "--load_best_model_at_end"
            ]
            if use_class_weights:
                argv.append("--use_class_weights")

            my_env = os.environ.copy()
            result = subprocess.run(['python'] + argv, capture_output=True, env=my_env)

            if result.returncode == 0:
                print('Training succeeded!')
            else:
                print('Training failed!')
                print((result.stdout).decode('utf-8'))
                print((result.stderr).decode('utf-8'))
                raise ValueError('Training failed!!')

            wandb.finish()

            log_file = sub_output_dir / "log.txt"
            with open(log_file, 'w') as f_log:
                f_log.write(f'n_splits: {n_splits}\n')
            log_file = sub_output_dir / "log.bin"
            with open(log_file, 'wb') as f_log:
                f_log.write(result.stdout)
                f_log.write(result.stderr)

    eval_data = reload_evaluation_results(output_dir, n_rerun, n_splits)

    return eval_data, output_dir
    
def reload_evaluation_results(output_dir, n_rerun, n_splits):
    """ Reload BERT evaluation results from an output run directory """
    eval_data = {i:{j:{} for j in range(n_rerun)} for i in range(n_splits)}
    for i in range(n_splits):
        for j in range(n_rerun):
            data_dir = pathlib.Path(output_dir) / f"split_{i}_run_{j}"

            model_dir = data_dir / "model"

            eval_file = model_dir / "eval_results_None.json"

            with open(eval_file, 'r') as f:
                eval_data_sub = json.load(f)
            eval_data[i][j] = eval_data_sub

    return eval_data    
