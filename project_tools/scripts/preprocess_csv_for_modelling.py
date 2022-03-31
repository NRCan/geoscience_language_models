# Copyright (C) 2021 ServiceNow, Inc.
# Preprocessing/cleaning script
# 
import nrcan_p2.data_processing.preprocessing_dfcol as preprocessing_dfcol
import nrcan_p2.data_processing.preprocessing_dfcol as preprocessing_str
import nrcan_p2.data_processing.pipelines as pipelines
from typing import Callable, List, Tuple, Dict
import pandas as pd
import argparse
import yaml
import pathlib
import json
import os 
from filelock import FileLock
import tqdm

KNOWN_PROCESSING_FUNCTIONS_FILE = 'PROCESSING_MAP.json'


def get_pipeline(pipeline_name):
    return pipelines.__getattribute__(pipeline_name), f'pipelines.{pipeline_name}'


def get_function_name(func):
    return func.__name__


def get_full_function_name(func):
    """ Return the full name of function, including the associated module 
        e.g. module.my_func() -> "module.my_func"
    """
    return f'{func.__module__}.{func.__name__}'


def get_function_module_name(func):
    """ Return the name of the module of a function
        e.g. module.my_funct() -> "module"
    """
    return func.__module__


def get_mapped_pipestep(func_name:str, code_to_func:Dict[str,int]):
    """ Given a mapping of function:id, return either the id associated 
        with this function or assign a new id 

    :param func_name: the name of the function to be mapped to an id
    :param code_to_func: the existing function:id mapping

    :returns: (the updated mapping dictionary, id of the function)
    """
    func_to_code = {v:int(k) for k,v in code_to_func.items()}

    # if we've already mapped this function
    if func_name in func_to_code:
        return code_to_func, func_to_code[func_name]

    # if the dict is empty...
    if len(func_to_code) == 0:
        new_key = 0
    else:    
        numeric_key_list = [int(x) for x in code_to_func.keys()]
        new_key = max(numeric_key_list) + 1
    code_to_func[new_key] = func_name
    return code_to_func, new_key


def get_existing_mapped_pipestep(func_name:str, code_to_func:Dict[str,int]):
    """ Given an existing mapping of function:id, return 
        the id associated with the function
        Will raise an error if the function does not already
        have an entry in the mapping

    :param func_name: the name of the function to be mapped
    :param code_to_func: the existing function:id mapping

    :returns: (the mapping dictionary, id of the function)
    """
    func_to_code = {v:int(k) for k,v in code_to_func.items()}

    # if we've already mapped this function
    if func_name in func_to_code:
        return code_to_func, func_to_code[func_name]   
    else:
        raise ValueError(f'{func_name} not already in mapping')


def print_and_log(s:str, fname:str, method='a'):
    """ Print to the console and also log to a log file
    """
    with FileLock(str(fname) + '.lock'):
        print(s)
        with open(fname, method) as f:
            if f is not None:
                f.write(s + '\n')
    

def count_files_in_folder(input_dir:str):
    """ Count the number of files and subdirectories in input_dir
    """
    total_files = 0
    for file in pathlib.Path(input_dir).iterdir():
        total_files += 1    
        
    return total_files


def build_next_temp_file_name(temp_file, orig_output_dir, new_output_dir, suffix, ext=".csv"):
    """ Given the name of the last preprocessing step's output file, 
        generate a new name for the ouptut file for the next step in the preprocessing pipeline
    """
    #temp_file = /../orig_output_dir/a/b/temp_file.csv
    
    # /a/b/temp_file.csv
    base_temp_file = temp_file.relative_to(orig_output_dir) 

    # new suffix
    # temp_file.other.csv OR temp_file.other__0__1.csv OR temp_file.csv or temp_file.0__1.csv
    ss_stem = pathlib.Path(base_temp_file.stem)

    # other__0__1
    ss_stem_suffix = ss_stem.suffix
    if ss_stem_suffix is None or ss_stem_suffix == "":
        ss_base = str(ss_stem) + "."
        old_suffix = ""
    else:
        ss = str(ss_stem.suffix).split('__') 
        ss_base = str(ss_stem.stem) + ss[0]
        old_suffix = '__'.join(ss[1:])

    assert old_suffix in str(temp_file)

    if old_suffix is not None and old_suffix != "":
        new_suffix = f"{old_suffix}__{suffix}"
    else:
        new_suffix = f"{suffix}"

    # /../new_output_dir/a/b/temp_file__suffix.csv
    new_temp_file = new_output_dir / new_suffix / base_temp_file.parent.relative_to(old_suffix) /  f"{ss_base}__{new_suffix}{ext}" #f"{base_temp_file.stem}{ext}"
    return new_temp_file


def process_pre_pipeline_step(col:str, 
                              next_col:str,
                              temp_file: pathlib.Path, 
                              next_temp_file:pathlib.Path, 
                              preprocessing_func: Callable[[pd.Series], pd.Series],
                              log_file=None,
                              p=None):
    """ Run a preprocessing step (preprocessing_func) on temp_file

    """
    # read existing file
    try:
        # Unfortunately, index_col fails on files with >1M lines
        df_per_pdf = pd.read_csv(temp_file, dtype={col: str}) #index_col=[0], dtype={'text': object})
    except Exception as e:
        print_and_log(f"Warning: exception occurred while reading {p}, {temp_file}. Skipping...", log_file)
        print_and_log(str(e), log_file)
        return False
                
    if col not in df_per_pdf.columns:
        print_and_log(f"Corrupted file found: {p}", log_file)
        return False

    # remove empty text
    df_per_pdf.dropna(subset=[col], inplace=True)

    if df_per_pdf.shape[0] == 0:
        print_and_log(f"Warning: empty file after null removal {p}. Skipping...", log_file)
        return False                 

    # run the next processing step
    preprocessing_func_module = get_function_module_name(preprocessing_func).split('.')[-1]
    if preprocessing_func_module == 'preprocessing_dfcol':
        try:
            df_per_pdf[next_col] = preprocessing_func(df_per_pdf[col])
        except Exception as e:
            print_and_log(f"Error occurred while processing {temp_file}.", log_file)
            print(df_per_pdf)
            raise(e)            
    elif preprocessing_func_module == 'preprocessing_df_filter':
        try: 
            df_per_pdf = preprocessing_func(df_per_pdf, col)
        except Exception as e:
            print_and_log(f"Error occurred while processing {temp_file}.", log_file)
            print(df_per_pdf)
            raise(e)  
    else:
        raise(ValueError("Unknown preprocessing module"))           

    # save it to file
    df_per_pdf.set_index(df_per_pdf.columns[0]).to_csv(next_temp_file)

    return True 


def process_dir(input_dir: str, 
                partial_output_dir: str,
                output_path: str, 
                preprocessing_pipe: List[Callable[[pd.Series],Tuple[pd.Series, str]]],
                postmerge_preprocessing_pipe: List[Callable[[pd.Series],Tuple[str, str]]],
                n_files: int,
                log_file: pathlib.Path,
                code_to_pipestep: dict,
                NO_FINAL_FILE:bool =False,
                PERC_FILE_START:int=0,
                PERC_FILE_END:int=100):

    """ Process all the csv files in input_dir through the pipeline
    """
      
    # [func ... ]

    preprocessing_pipe_funcs, pipename = get_pipeline(preprocessing_pipe)
    print_and_log(f"Will run pipe {pipename}", log_file)
    postprocessing_pipe_funcs, post_pipename = get_pipeline(postmerge_preprocessing_pipe)
    print_and_log(f"Followed by pipe {post_pipename}", log_file)
    
    # find all files
    total_files = count_files_in_folder(input_dir)
            
    i_file = 0
    processed_files = []
    
    print_and_log(f"Reading files in {input_dir}", log_file)

    input_dir = pathlib.Path(input_dir)
    partial_output_dir = pathlib.Path(partial_output_dir)
    output_path = pathlib.Path(output_path)

    if PERC_FILE_START is not None:
        if not NO_FINAL_FILE:
            raise ValueError('Cannot request a PERC_FILE_START and NO_FINAL_FILE')
        start_file_i = int(total_files * PERC_FILE_START/100.0)
        end_file_i = int(total_files * PERC_FILE_END/100.0)
    else:
        start_file_i = 0
        end_file_i = total_files
    print_and_log(f"...reading between {start_file_i} and {end_file_i}", log_file)

    file_i = -1

    for p in tqdm.tqdm(pathlib.Path(input_dir).iterdir(), total=total_files):
        file_i += 1

        if file_i < start_file_i or file_i > end_file_i:
            continue

        # assume there's no sub directory in dir
        if p.suffix != ".csv":
            continue

        temp_file = p
            
        did_process = True
        for i, preprocessing_func in enumerate(preprocessing_pipe_funcs):
            pipestep = get_function_name(preprocessing_func)
            code_to_pipestep, pipestep_mapped = get_existing_mapped_pipestep(get_full_function_name(preprocessing_func), code_to_pipestep)

            if i == 0:
                next_temp_file = build_next_temp_file_name(temp_file, input_dir, partial_output_dir, suffix=pipestep_mapped, ext=".csv") 
                col = "text"
            else:
                next_temp_file = build_next_temp_file_name(temp_file, partial_output_dir, partial_output_dir, suffix=pipestep_mapped, ext=".csv")
                col = "processed_text"

            next_temp_file.parent.mkdir(parents=True, exist_ok=True)
            
            next_col = "processed_text"
                
            # skip this step if already generated
            if next_temp_file.exists():
                temp_file = next_temp_file
                continue
        
            did_process = process_pre_pipeline_step(col, next_col, temp_file, next_temp_file, preprocessing_func, log_file, p)
            if not did_process:
                break

            # reset loop
            temp_file = next_temp_file
        
        # this file can't be processed, so bail now
        if not did_process:
            print_and_log(f'Completely skipping file... {p}', log_file)
            continue

        # remove \n so all boxes from the same pdf will be combined into one doc
        next_temp_file = build_next_temp_file_name(temp_file, partial_output_dir, partial_output_dir, suffix="POST", ext=".txt")
        next_temp_file.parent.mkdir(parents=True, exist_ok=True)
        if next_temp_file.exists():
            with open(next_temp_file, 'r') as f:
                text = f.read()
        else:
            df_per_pdf = pd.read_csv(temp_file, dtype={next_col:"str"})
            text = " ".join(df_per_pdf[next_col].tolist())
            with open(next_temp_file, 'w') as f:
                f.write(text)

        temp_file = next_temp_file

        for i, postprocessing_func in enumerate(postprocessing_pipe_funcs):

            pipestep = get_function_name(postprocessing_func)
            code_to_pipestep, pipestep_mapped = get_existing_mapped_pipestep(get_full_function_name(postprocessing_func), code_to_pipestep)
            next_temp_file = build_next_temp_file_name(temp_file, partial_output_dir, partial_output_dir, suffix=pipestep_mapped, ext='.txt')
            next_temp_file.parent.mkdir(parents=True, exist_ok=True)

            if next_temp_file.exists():
                with open(next_temp_file, 'r') as f:
                    text = f.read()
                
                temp_file = next_temp_file
                continue

            text = postprocessing_func(text)

            with open(next_temp_file, 'w') as out_f:
                out_f.write(text)

            temp_file = next_temp_file

        if not NO_FINAL_FILE:
            # write everything to the final file!
            with open(output_path, "a") as out_f:
                out_f.write(text + "\n")

        i_file += 1
        processed_files.append(str(p))
        if n_files is not None and n_files != -1:
            if i_file >= n_files:
                break

    print_and_log(f"Total number of docs written: {i_file}", log_file)
    return processed_files
       

def write_config_file(config_file, args, output_file, pipestep_mapping_file):
    preprocessing_pipe_funcs, pipename = get_pipeline(args.PREPROCESSING_PIPELINE)
    postprocessing_pipe_funcs, post_pipename = get_pipeline(args.POST_PIPELINE)

    pipesteps = []
    pipesteps_mapped = []

    with FileLock(str(pipestep_mapping_file) + ".lock"):

        with open(pipestep_mapping_file, 'r') as f_sm:
            code_to_pipestep = json.load(f_sm)

        for preprocessing_func in preprocessing_pipe_funcs:
            pipestep = get_function_name(preprocessing_func)
            code_to_pipestep, pipestep_mapped = get_mapped_pipestep(get_full_function_name(preprocessing_func), code_to_pipestep)
            pipesteps.append(pipestep)
            pipesteps_mapped.append(pipestep_mapped) 

        postpipesteps = []
        postpipesteps_mapped = []
        for postprocessing_func in postprocessing_pipe_funcs:
            pipestep = get_function_name(postprocessing_func)
            code_to_pipestep, pipestep_mapped = get_mapped_pipestep(get_full_function_name(postprocessing_func), code_to_pipestep)
            postpipesteps.append(pipestep)
            postpipesteps_mapped.append(pipestep_mapped)

        with open(pipestep_mapping_file, 'w') as f_sm:
            json.dump(code_to_pipestep, f_sm, indent=4)               

    config = {
        'preprocessing_pipeline': args.PREPROCESSING_PIPELINE,
        'preprocessing_functions': pipesteps,
        'preprocessing_functions_mapped': pipesteps_mapped,
        'postprocessing_pipeline': args.POST_PIPELINE,
        'postprocessing_functions': postpipesteps,
        'postprocessing_functions_mapped': postpipesteps_mapped,        
        'input_dirs': args.INPUT_DIRS,
        'output_dir': args.OUTPUT_DIR,
        'n_files': args.N_FILES,
        'suffix': args.SUFFIX,
        'output_file': str(output_file)
    }
    print(f"Writing config to {config_file}")
    with FileLock(str(config_file) + ".lock"):
        with open(config_file, 'w') as f:
            yaml.dump(config, f)

    return code_to_pipestep


def main(args):

    print(f"Number of dirs to process: {len(args.INPUT_DIRS)}")
    print(args.INPUT_DIRS)

    print(args.NO_FINAL_FILE)

    # ensure output dir is a directory
    if not os.path.isdir(args.OUTPUT_DIR):
        raise ValueError(f"Not a directory (output dir): {args.OUTPUT_DIR}")
                
    if not os.path.isdir(args.PARTIAL_OUTPUT_DIR):
        raise ValueError(f"Not a directory (partial output dir): {args.PARTIAL_OUTPUT_DIR}")

    step_mapping_file = pathlib.Path(args.PARTIAL_OUTPUT_DIR) / KNOWN_PROCESSING_FUNCTIONS_FILE

    with FileLock(str(step_mapping_file) + '.lock'):
        if not os.path.exists(step_mapping_file):
            print(f'Creating step mapping {step_mapping_file}')
            step_mapping = {}
            with open(step_mapping_file, 'w') as f:
                json.dump(step_mapping, f, indent=4)       

        else:
            print(f'Reading step mapping from file {step_mapping_file}')
            with open(step_mapping_file, 'r') as f:
                step_mapping = json.load(f)
        
        print('Existing step mapping:')
        print(step_mapping)

                
    if args.PERC_FILE_END == -1:
        args.PERC_FILE_END = None
    if args.PERC_FILE_START == -1:
        args.PERC_FILE_START = None
    SUFFIX = args.SUFFIX            
    if args.N_FILES and args.N_FILES != -1:
        SUFFIX = f"{SUFFIX}_{args.N_FILES}"
                
    # create file names
    output_path = pathlib.Path(args.OUTPUT_DIR) / f"all_text_{SUFFIX}.txt"
    log_file = pathlib.Path(args.OUTPUT_DIR) / f"all_text_{SUFFIX}.log"
    source_file = pathlib.Path(args.OUTPUT_DIR) / f"all_text_{SUFFIX}_source.csv"
    config_file = pathlib.Path(args.OUTPUT_DIR) / f"all_text_{SUFFIX}.config"    
        
    # ensure output file does not already exist
    print(f'Will write output to {output_path}')
    if os.path.exists(output_path):
        raise ValueError(f"file already exists: {output_path}")
                
    # ensure all input dirs exist
    for input_dir in args.INPUT_DIRS:
        if not os.path.isdir(input_dir):
            raise ValueError(f"Not a directory: {input_dir}")

    # write config file
    step_mapping = write_config_file(config_file, args, output_path, step_mapping_file)

    print(f'New step mapping:')
    print(step_mapping)
      
    #with open(log_file, "w") as log_f:
    processed_files_all = []
    for input_dir in args.INPUT_DIRS:
        processed_files = process_dir(
            input_dir=input_dir, 
            partial_output_dir=args.PARTIAL_OUTPUT_DIR,
            output_path=output_path,
            preprocessing_pipe=args.PREPROCESSING_PIPELINE,
            postmerge_preprocessing_pipe=args.POST_PIPELINE,
            n_files=args.N_FILES,
            log_file=log_file,
            code_to_pipestep=step_mapping, 
            NO_FINAL_FILE=args.NO_FINAL_FILE,
            PERC_FILE_START=args.PERC_FILE_START,
            PERC_FILE_END=args.PERC_FILE_END)    
        processed_files_all.extend(processed_files)
        
    # write processed files to source
    print(f"Writing source_file list to {source_file}")
    df_source = pd.DataFrame({'file': processed_files_all})
    print(f"Total number of docs (all inputs) written: {df_source.shape[0]}")
    df_source.to_csv(source_file, index=None)
    
    
def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    requiredNamed = parser.add_argument_group('required named arguments')
    requiredNamed.add_argument('--INPUT_DIRS', nargs="+", help='list of directories of input csvs')
    requiredNamed.add_argument('--PARTIAL_OUTPUT_DIR', help='output directory of the partial csvs')
    requiredNamed.add_argument('--OUTPUT_DIR', help='output directory of the combined text')
    requiredNamed.add_argument('--PREPROCESSING_PIPELINE', help='preprocessor_function from nrcan_2.preprocessing')
    requiredNamed.add_argument('--POST_PIPELINE', help="sentence tokenizer")
    requiredNamed.add_argument('--SUFFIX', help='suffix to be appended to all filenames')
    requiredNamed.add_argument('--NO_FINAL_FILE', type=str2bool, help='whether or not this is a partial run')
    parser.add_argument('--N_FILES', type=int,
                        help='maximum number of files to process (for debugging, -1 or not present to ignore')
    parser.add_argument('--PERC_FILE_START', type=int, help='perc of files at which to start reviewing')
    parser.add_argument('--PERC_FILE_END', type=int, help='perc of file at which to end reviewing')
    args = parser.parse_args()

    main(args)    
    