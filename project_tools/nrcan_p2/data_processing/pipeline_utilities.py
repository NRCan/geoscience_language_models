# Copyright (C) 2021 ServiceNow, Inc.
""" Pipeline utilities 

    Functionality for operating with pipelines.
"""
import nrcan_p2.data_processing.pipelines as pipelines


def get_function_name(func):
    """ Return the name of a function """
    return func.__name__


def print_and_log(s:str, fname:str, method='a'):
    """ Print to the console and also log to a log file
    """
    if fname is None:
        print(s)
        return

    with FileLock(str(fname) + '.lock'):
        print(s)
        with open(fname, method) as f:
            if f is not None:
                f.write(s + '\n')


def get_function_module_name(func):
    """ Return the name of the module of a function
        e.g. module.my_funct() -> "module"
    """
    return func.__module__


def get_pipeline(pipeline_name):
    """ Return the pipeline object from data_processing.pipelines.py given its name """
    return pipelines.__getattribute__(pipeline_name), f'pipelines.{pipeline_name}'


def run_pipeline(df, col, next_col, preprocessing_pipe=None, postmerge_preprocessing_pipe=None, log_file=None):
    """ Given an input df, with a column of text, apply the preprocessin_pipe and 
        postmerge_preprocessing_pipe. However, apply each step in the pipeline to each row 
        independently (ie never merge across rows). The dataframe will be updated 

    :param df: input dataframe
    :param col: text column name
    :param next_col: the column name for the output processed text
    :param preprocessing_pipe: name of the preprocessing pipeline to apply
    :param postmerge_preprocessing_pipe: name of preprocessing postmerge pipeline to apply
    :param log_file: fileobject for logging
    """
    if preprocessing_pipe is not None:
        preprocessing_pipe_funcs, pipename = get_pipeline(preprocessing_pipe)
    if postmerge_preprocessing_pipe is not None:
        postprocessing_pipe_funcs, post_pipename = get_pipeline(postmerge_preprocessing_pipe)

    if preprocessing_pipe is not None:
        for i, preprocessing_func in enumerate(preprocessing_pipe_funcs):

            pipestep = get_function_name(preprocessing_func)
            preprocessing_func_module = get_function_module_name(preprocessing_func).split('.')[-1]

            if preprocessing_func_module == 'preprocessing_dfcol':
                try:
                    df[next_col] = preprocessing_func(df[col])
                except Exception as e:
                    print_and_log(f"Error occurred while processing .", log_file)
                    print(df)
                    raise(e)            
            elif preprocessing_func_module == 'preprocessing_df_filter':
                try: 
                    df = preprocessing_func(df, col)
                except Exception as e:
                    print_and_log(f"Error occurred while processing .", log_file)
                    print(df)
                    raise(e)  
            else:
                raise(ValueError("Unknown preprocessing module"))             

            df = df.dropna(subset=[next_col])

    if postmerge_preprocessing_pipe is not None:
        text = " ".join(df[next_col].tolist())

        for i, postprocessing_func in enumerate(postprocessing_pipe_funcs):
            pipestep = get_function_name(postprocessing_func)
            
            text = postprocessing_func(text)

        return text
    else:
        return df