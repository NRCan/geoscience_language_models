# Copyright (C) 2021 ServiceNow, Inc.
""" Compile results from bert keyword evaluation 
    This script compiles the MULTICLASS and PAIRING evaluation output 
    files in the input_dir into the two files specified by 
    OUTPUT_GLOVE_RUN_LOG and OUTPUT_GLOVE_RUN_METRICS which are 
    used by the evaluation notebooks.
"""

import pathlib
import datetime 
import pandas as pd
import json
import re

input_dir = pathlib.Path('/nrcan_p2/data/07_model_output/keyword_prediction_glove/')

OUTPUT_GLOVE_RUN_LOG = 'glove_runs_log.csv'
OUTPUT_GLOVE_RUN_METRICS = 'glove_eval_data.parquet'

######################################

run_dirs = [x for x in input_dir.iterdir()]

legal_runs = []
illegal_or_unfinshed_runs = []
input_info = []
eval_datas = []
for run_dir in run_dirs: 
    is_illegal = False

    run_time = datetime.datetime.strptime(run_dir.name.split('_')[1], "%Y-%m-%d-%H-%M-%S%f") #"2021-03-04-00-30-27382488"

    input_log_file = run_dir / 'input_data.log'
    if not input_log_file.exists():
        is_illegal = True

    else:
        i_sets = [(i,j) for i in range(0,5) for j in range(0,3)]

        for i_set in i_sets:
            split_dir = run_dir / f'split_{i_set[0]}_run_{i_set[1]}'
            if not split_dir.exists():
                is_illegal = True
                break

            metrics_file = split_dir / "metrics.json"
            model_file = split_dir / "model.joblib"
            model_params_file = split_dir / "model_params.json" 
            if not metrics_file.exists() or not model_file.exists() or not model_params_file.exists():
                is_illegal = True
                break

            
            with open(metrics_file, 'r') as f:
                eval_data = json.load(f)

            eval_data = pd.DataFrame([eval_data])
            eval_data['split'] = i_set[0]
            eval_data['nrun'] = i_set[1]  
            eval_data['run_dir'] = str(run_dir)
            eval_datas.append(eval_data)       

    if is_illegal:
        illegal_or_unfinshed_runs.append((run_time,run_dir))

    else:
        legal_runs.append((run_time,run_dir))

        with open (input_log_file, 'r') as f:
            input_log = json.load(f)

        input_info.append(input_log)


print(len(illegal_or_unfinshed_runs))
print(len(legal_runs))

illegal_or_unfinished_runs = pd.DataFrame(illegal_or_unfinshed_runs, columns=['date', 'path'])
illegal_or_unfinished_runs['finished'] = False
legal_runs = pd.DataFrame(legal_runs, columns=['date', 'path'])
legal_runs['finished'] = True
input_info = pd.DataFrame(input_info)

concat_info = pd.concat([legal_runs, input_info], axis=1)

def parse_dataset_name(s):
    #/nrcan_p2/data/03_primary/keyword_prediction/splits/MULTICLASS_small_subject_5_title_merged_SIMPLE_PIPELINE_GLOVE_3_POSTPIPE_GLOVE_nodrop-Feb29
    m = re.search("((MULTICLASS)|(PAIRING))_small_((subject_5)|(subject_30)|(subject_desc_t10))_((title_merged)|(desc_en_en_50_3000)|(desc_en_en))_((N|P|S).+)_(nNone_)?nodrop-Feb29", s)
    if m is None:
        print(s)
    task = m.group(1)
    subject = m.group(4)
    text = m.group(8)
    pipeline = m.group(12)

    return task, subject, text, pipeline

concat_info['data_infos'] = concat_info.data_dir.apply(lambda x: parse_dataset_name(x))
concat_info['data_task'] = concat_info.data_infos.str[0]
concat_info['data_subject'] = concat_info.data_infos.str[1]
concat_info['data_text'] = concat_info.data_infos.str[2]
concat_info['data_pipeline'] = concat_info.data_infos.str[3]

concat_info = pd.concat([concat_info, illegal_or_unfinished_runs], axis=0)

concat_info.to_csv(OUTPUT_GLOVE_RUN_LOG)


eval_datas = pd.concat(eval_datas)
print(len(eval_datas))
eval_datas.to_parquet(OUTPUT_GLOVE_RUN_METRICS)
         

                
                