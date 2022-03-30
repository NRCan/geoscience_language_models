# Data Processing

## 1. PDF -> csv conversion
----------------------------
The first step is to convert pdfs to the internal .csv format that is used by the data processing pipelines. The current scripts will not attempt to reprocess files for which an corresponding .csv is already found in the output folder. 

### Running locally
--------------------
Run the following script to convert all pdf files in a given folder specified by `--LOCAL_DIR` (replace the sample parameters with your own):
```
python scripts/pdf_to_txt.py --N_FILES -1 --WRITE_OUTPUT --LOCAL_DIR /nrcan_p2/data/01_raw/20201006/geoscan/raw/pdf --OUTPUT_DIR /nrcan_p2/data/02_intermediate/20201006/geoscan/pdf/v1_20201125 
```
See `python scripts/pdf_to_txt.py --help` for other options to convert subsets of the files.

**NOTE**: You may wish to update the `TIMEOUT` parameter which controlls the amount of time a document is given
before the conversion is aborted. This is a hardcoded parameter in the `scripts/pdf_to_txt.py` script.

## 2. Preprocessing/Cleaning 
----------------------------
The second step in preprocessing is to pass the output .csv from the PDF->csv conversion through the cleaning pipelines. It makes use of the preprocessing pipelines defined in `nrcan_p2/data_preprocessing/pipelines.py`

#### Preprocessing pipelines
----------------------------
The preprocessing pipelines are defined in `nrcan_p2/data_preprocessing/pipelines.py`. In the cleaning scripts, pipelines are referred to by their string names (e.g. "PIPELINE_GLOVE_80"). See the file for the full list. 

There are two kinds of pipelines: "preprocessing" and "postprocessing":
* **Preprocessing pipelines**: consist of pipeline steps that operate on .csv files, under the assumption that each row in the .csv corresponds to a textbox and each .csv corresponds to a pdf document
* **Postprocessing pipelines**: consist of pipeline steps that operate on .txt files, under the assumption that each paragraph in the .txt corresponds to a paragraph and the .txt as a whole corresponds to a single pdf document

You must select one of each type of pipeline and pass their names to the scripts outlined below.

#### Pipeline process
---------------------
The dataset cleaning script proceeds as follows:  
1) read in all .csv files from the input .csv folder (or folders)
2) apply every step of the specified **preprocessing pipeline** to every .csv file
    * intermediary output .csv files for each step in the pipeline will be saved in a different subfolder for each step
3) merge the text in each .csv into one .txt file per .csv and save these in another subfolder
4) apply every step of the specified **postprocessing pipeline** to every .txt file
    * intermediary output .txt files for each step in the pipeline will be saved in a different subfolder for each step
5) combine the .txt files into one final .txt dataset file

Each step in the pipeline will be automatically associated with a unique ID. This ID is maintained across runs of the pipeline. The mapping is stored in `PROCESSING_MAP.json` in the output folder. If you switch output folders, a new (and not necessarily identical) map will be generated. 

The subfolders that contain the intermediate output files for each step are named according to the step ID mentioned above. For example, if our full pipeline has 3 preprocessing steps and 2 post processing steps, then we will will have the following intermediate output folders:
```
0
0_1
0_1_2
0_1_2_POST
0_1_2_POST_3
0_1_2_POST_3_4
```

the `0_1_2_POST` folder contains .txt versions of the .csv files from the previous step, without any postprocessing steps applied. While in this example, the pipeline steps correspond to consecutive numbers, this is not necessarily the case in practice. 

The names of the individual files are preserved through each pipeline step. These names *must* be unique. 

Note that this script operates on *every .csv file it can find* in the input folders. This means that if the number of files in the input changes, then the output dataset will also differ. 

#### Output files
-----------------
The pipeline generation process outputs a number of files: 
* `PROCESSING_MAP.json`: the pipeline step to ID map, described above
* `all_text_{PREPROCESSING_PIPELINE}_{POSTPROCESING_PIPELINE}_{DATASET}_{SUFFIX}.txt`: the final output dataset name
* `all_text_{PREPROCESSING_PIPELINE}_{POSTPROCESING_PIPELINE}_{DATASET}_{SUFFIX}.config`: json formatted file that describes the exact pipeline steps and settings used to produce the dataset file
* `all_text_{PREPROCESSING_PIPELINE}_{POSTPROCESING_PIPELINE}_{DATASET}_{SUFFIX}.log`: log file from data processing
* `all_text_{PREPROCESSING_PIPELINE}_{POSTPROCESING_PIPELINE}_{DATASET}_{SUFFIX}.source`: csv containing the names of all of the .txt files from the final postprocessing pipeline step before they were merged into the final output dataset file.

Note that the script will not proceed if the specified output file already exists. If you want to regenerate a given dataset, you must first delete the output.txt file. 

### Running locally
-------------------
To run preprocessing locally, run the `scripts/preprocess_csv_for_modelling.py` script. 

To run on *all* .csv files in a set of folders: 
```
python scripts/preprocess_csv_for_modelling.py 
    --INPUT_DIRS /path/to/csv/folder /another/path/to/csv/folder
    --PARTIAL_OUTPUT_DIR /path/to/output/folder
    --OUTPUT_DIR /path/to/output/folder
    --PREPROCESSING_PIPELINE preprocessing_pipe_name
    --POST_PIPELINE post_processing_pipe_name
    --SUFFIX v1
    --NO_FINAL_FILE False
    --N_FILES -1
```
It is best to set the `PARTIAL_OUTPUT_DIR` to match the `OUTPUT_DIR`. All intermediate subfolders and the final output folders will be saved in `PARTIAL_OUTPUT_DIR` and `OUTPUT_DIR` respectively. Any distinction between the input folders will be lost. All files must thus have unique names.

To run on a subset of the .csv files (e.g. the 5th-10th percent of the total list of files):
```
python scripts/preprocess_csv_for_modelling.py 
    --INPUT_DIRS /path/to/csv/folder /another/path/to/csv/folder
    --PARTIAL_OUTPUT_DIR /path/to/output/folder
    --OUTPUT_DIR /path/to/output/folder
    --PREPROCESSING_PIPELINE preprocessing_pipe_name
    --POST_PIPELINE post_processing_pipe_name
    --SUFFIX v1
    --NO_FINAL_FILE True
    --N_FILES -1
    --PERC_FILE_START 0
    --PERC_FILE_END 95
```
This allows you split the processing work across a number of different processes. If `--NO_FINAL_FILE` is True, then the output files will all have the `_partial` suffix appended before their extensions. No final `.txt` file will be generated in this case and the `.source` file should not be trusted. 

If `--N_FILES` is set to anything other than "-1" that number of files will be processed instead of the full set (or subset) in the input folders. In this case, the output files will have `_{N}` appended to the end of their filenames, where N is the specified number to process *per input folder*. 


## 3. Combining datasets
------------------------
After producing preprocessed datasets for training, you may wish to combine multiple datasets into a single trianing dataset file (e.g. combine datasets A and B into A+B). 

Update the parameters in the following script to point to your dataset and run the following command on a machine that has access to the data file location:

```
python scripts/combine_dataset.py 
```
