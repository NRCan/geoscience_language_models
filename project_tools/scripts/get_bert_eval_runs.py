# Copyright (C) 2021 ServiceNow, Inc.
""" Compile results from bert keyword evaluation 
    This script compiles the MULTICLASS and PAIRING evaluation output 
    files in the INPUT_FOLDER into the two files specified by 
    OUTPUT_FILE_EVAL_DATA and OUTPUT_FILE_EVAL_METRICS which are 
    used by the evaluation notebooks.
"""

import re 
import pathlib
import json
import pandas as pd

INPUT_FOLDER = '/nrcan_p2/data/07_model_output/keyword_prediction_bert/'

OUTPUT_FILE_EVAL_DATA = 'bert_eval_data.csv'
OUTPUT_FILE_EVAL_METRICS = 'bert_eval_metrics.parquet'

######################################

INPUT_FOLDER = pathlib.Path(INPUT_FOLDER)
print(INPUT_FOLDER)

dfs = []
eval_datas = []
for rundir in pathlib.Path(INPUT_FOLDER).iterdir():
    #print(rundir)

    if not 'MULTICLASS' in str(rundir) and not 'PAIRING' in str(rundir):
        continue
    
    input_data_file = rundir / 'input_data.log'
    with open(input_data_file) as f:
        input_data = f.read().strip()

    if input_data == "":
        dataset = None
        subject_col = None
        text_col = None
        pipeline = None
        model_path = None
        use_class_weights = None
        finished = False
        nodrop = None

    else:
        if 'MULTICLASS' in str(input_data):
            task = 'MULTICLASS'
        elif 'PAIRING' in str(input_data):
            task = 'PAIRING'
        else:
            raise ValueError(f'Unknown input file {input_data}')

        if 'small' in str(input_data):
            dataset = 'small'
        elif 'large' in str(input_data):
            dataset = 'large'
        else:
            raise ValueError(f'Unknown input file {input_data}')

        subject_col = None
        if 'subject_5' in str(input_data):
            subject_col = 'subject_5'
        elif 'subject_30' in str(input_data):
            subject_col = 'subject_30'
        elif 'subject_desc_t10' in str(input_data):
            subject_col = 'subject_desc_t10'
        elif 'subject_g200' in str(input_data):
            subject_col = 'subject_g200'
        else:
            raise ValueError(f'Unknown input file {input_data}')

        
        if 'desc_en_en_50_3000' in str(input_data):
            text_col = 'desc_en_en_50_3000'
        elif 'desc_en_en' in str(input_data):
            text_col = 'desc_en_en'
        elif 'title_merged' in str(input_data):
            text_col = 'title_merged'
        else:
            raise ValueError(f'Unknown input file {input_data}')

        if 'nodrop' in str(input_data):
            nodrop=True
        else:
            nodrop=False

        pipeline = None
        if nodrop:
            if 'SIMPLE_PIPELINE_BERT_3_None' in str(input_data):
                pipeline = 'SIMPLE_PIPELINE_BERT_3_None'
            elif 'None_None_nodrop' in str(input_data):
                pipeline = 'None_None'
            elif 'PIPELINE_BERT_80_None_nodrop' in str(input_data):
                pipeline = 'PIPELINE_BERT_80_None'
            elif 'PIPELINE_BERT_90_None_nodrop' in str(input_data):
                pipeline = 'PIPELINE_BERT_90_None'
            elif 'PIPELINE_BERT_PLUS_None_nodrop' in str(input_data):
                pipeline = 'PIPELINE_BERT_PLUS_None'
            else:
                raise ValueError(f'Unknown input file {input_data}')

        
        model_args_file = rundir / 'split_0_run_0' / 'model' / 'model_args.json'
        use_class_weights = None
        finished = False
        model_path = None
        if (model_args_file).exists():
            with open(model_args_file) as f:
                model_args_data = f.read().strip()

            m = re.search('model_name_or_path=\'([^,)\']+)\'[,)]', model_args_data)
            model_path = m.group(1)

            m = re.search('use_class_weights=\'?([^,)\']+)\'?[,)]', model_args_data)
            if m:
                use_class_weights = m.group(1)
            else:
                use_class_weights = None

            finished = True
            for i in range(0,5):
                for j in range(0,3):
                    if not (rundir / f'split_{i}_run_{j}' / 'model' / 'eval_results_None.txt').exists():
                        finished = False

                    else:
                        eval_results_file = rundir / f'split_{i}_run_{j}' / 'model' / 'eval_results_None.json'
                        with open(eval_results_file) as f:
                            eval_results = json.load(f)
                        
                        eval_data = pd.DataFrame([eval_results])
                        eval_data['split'] = i
                        eval_data['nrun'] = j  
                        eval_data['run_dir'] = str(rundir.name)
                        eval_data['run_dir_full'] = str(rundir)
                        eval_data['train/eval'] = 'eval'
                        eval_datas.append(eval_data)  

                        train_results_file = rundir / f'split_{i}_run_{j}' / 'model' / 'train_results.json'
                        with open(train_results_file) as f:
                            train_results = json.load(f)
                        
                        train_data = pd.DataFrame([train_results])
                        train_data['split'] = i
                        train_data['nrun'] = j  
                        train_data['run_dir'] = str(rundir.name)
                        train_data['run_dir_full'] = str(rundir)
                        train_data['train/eval'] = 'train'
                        eval_datas.append(train_data)                          

    df = pd.DataFrame({
        'task': task,
        'rundir': str(rundir.name),
        'run_dir_full': str(rundir),
        'dataset': dataset,
        'subject': subject_col,
        'text_col': text_col,
        'meta_pipeline': pipeline,
        'bert_model': model_path,
        'use_class_weights': use_class_weights,
        'finished': finished,
        'nodrop': nodrop
    }, index=[0])

    dfs.append(df)

dfs = pd.concat(dfs)

### Write the output files
dfs.to_csv(OUTPUT_FILE_EVAL_DATA)

eval_datas = pd.concat(eval_datas)
eval_datas.to_parquet(OUTPUT_FILE_EVAL_METRICS)
    
