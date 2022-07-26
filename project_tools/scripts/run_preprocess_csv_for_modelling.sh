# Copyright (C) 2021 ServiceNow, Inc.
#
# the preferred alternative to run_preprocess_csv_for_modelling.py
#

# Set to a positive number to process only N files per input folder
# (for debugging)
N_FILES="-1"

# Set below according to the provided dataset name
INPUT_DIRS=
OUTPUT_DIR=
PARTIAL_OUTPUT_DIR=

# Dataset name
DATASET="B" #A_full, B, D

## Preprocessing pipeline options
#PRE_PIPELINE='SIMPLE_PIPELINE_BERT_3'
#PRE_PIPELINE='SIMPLE_PIPELINE_GLOVE_3'
#PRE_PIPELINE='PIPELINE_GLOVE_90'
#PRE_PIPELINE='PIPELINE_GLOVE_80'
#PRE_PIPELINE='PIPELINE_GLOVE_PLUS'
#PRE_PIPELINE='PIPELINE_BERT_80'
#PRE_PIPELINE='PIPELINE_BERT_90'
PRE_PIPELINE='PIPELINE_BERT_PLUS'

## Postprocessing pipeline options
POST_PIPELINE='POSTPIPE_BERT_SPACY_2'
#POST_PIPELINE='POSTPIPE_GLOVE'

# Set to True to run this script on multiple processes for subsets
# of the input files
IS_PARTIAL="False"

# Run from END-START percentiles of the files (100-0 for all files)
# with a different process for every STEP percent 
START=100
END=0
STEP=5

#################################
SUFFIX="${PRE_PIPELINE}_${POST_PIPELINE}_d${DATASET}_v1"


if [ "${DATASET}" = "A_full" ]
then
    echo "Dataset A..."
    INPUT_DIRS=(
        '/nrcan_p2/data/02_intermediate/20201006/geoscan/pdf/v1_all'
        '/nrcan_p2/data/02_intermediate/20201117/geoscan/pdf/generic_pdfs_all'
        '/nrcan_p2/data/02_intermediate/20201117/geoscan/pdf/has_pdf_dir_all'
        '/nrcan_p2/data/02_intermediate/20201117/geoscan/pdf/of_pdf_all'
        '/nrcan_p2/data/02_intermediate/20201117/geoscan/pdf/low_text_pdfs'
        '/nrcan_p2/data/02_intermediate/20201117/geoscan/txt'
        '/nrcan_p2/data/02_intermediate/20201117/geoscan/wp_rtf'    
    )
    OUTPUT_DIR='/nrcan_p2/data/03_primary/v4'
    PARTIAL_OUTPUT_DIR='/nrcan_p2/data/03_primary/v4'    
elif [ "${DATASET}" = "A" ]
then
    echo "Dataset A..."
    INPUT_DIRS=(
        '/nrcan_p2/data/02_intermediate/20201006/geoscan/pdf/v1_all'
        '/nrcan_p2/data/02_intermediate/20201117/geoscan/pdf/generic_pdfs_all'
        '/nrcan_p2/data/02_intermediate/20201117/geoscan/pdf/has_pdf_dir_all'
        '/nrcan_p2/data/02_intermediate/20201117/geoscan/pdf/of_pdf_all'
        '/nrcan_p2/data/02_intermediate/20201117/geoscan/pdf/low_text_pdfs' 
    )
    OUTPUT_DIR='/nrcan_p2/data/03_primary/v4'
    PARTIAL_OUTPUT_DIR='/nrcan_p2/data/03_primary/v4'        
elif [ "${DATASET}" = "B" ]
then
    echo "Dataset B..."
    INPUT_DIRS=(
        '/nrcan_p2/data/02_intermediate/20210108'
    )
    OUTPUT_DIR='/nrcan_p2/data/03_primary/v4_B'
    PARTIAL_OUTPUT_DIR='/nrcan_p2/data/03_primary/v4_B'      
elif [ "${DATASET}" = "D" ]
then 
    echo "Dataset D..."
    INPUT_DIRS=(
        '/nrcan_p2/data/02_intermediate/20201221/doaj'
    )
    OUTPUT_DIR='/nrcan_p2/data/03_primary/v4_D'
    PARTIAL_OUTPUT_DIR='/nrcan_p2/data/03_primary/v4_D'
fi

echo "Running..."
echo ${INPUT_DIRS[@]}
echo "${DATASET}"

REAL_END=$(($END+$STEP))
echo ${REAL_END}

if [ "${IS_PARTIAL}" = "True" ]
then
SUFFIX="${SUFFIX}_partial"
fi

NOW=$(date +"%m_%d_%Y_%H_%M_%S")

echo ${SUFFIX}
SUFFIX_LOWER=$(echo "${SUFFIX}" | tr '[:upper:]' '[:lower:]')
DATASET_LOWER=$(echo "${DATASET}" | tr '[:upper:]' '[:lower:]')
echo ${SUFFIX_LOWER}

make build
make push

if [ "${IS_PARTIAL}" = "True" ]
then 
    partial_folder_i=0
    for partial_folder in "${INPUT_DIRS[@]}"
    do
        for perc_end in $(seq ${START} -${STEP} ${REAL_END})
        do
            echo $perc_end
            perc_start=$(($perc_end-${STEP}))
            echo $perc_start
            
            CMD="python scripts/preprocess_csv_for_modelling.py --INPUT_DIRS ${partial_folder} --N_FILES ${N_FILES} --PARTIAL_OUTPUT_DIR ${PARTIAL_OUTPUT_DIR} --OUTPUT_DIR ${OUTPUT_DIR} --PREPROCESSING_PIPELINE ${PRE_PIPELINE} --POST_PIPELINE ${POST_PIPELINE} --SUFFIX ${SUFFIX} --NO_FINAL_FILE ${IS_PARTIAL} --PERC_FILE_START ${perc_start} --PERC_FILE_END ${perc_end}"
            echo ${CMD}

            preprocess_job_name="preprocess_d${DATASET_LOWER}_${SUFFIX_LOWER}_${partial_folder_i}_${perc_start}_${NOW}"
            echo ${preprocess_job_name}
            echo ${ENV_ARR[@]}
	    echo "Error: Internal job management tool was to execute the command. Please run the command locally."
        done
        partial_folder_i=$((partial_folder_i+1))
    done
else
    perc_start=-1
    perc_end=-1
    CMD="python scripts/preprocess_csv_for_modelling.py --INPUT_DIRS ${INPUT_DIRS[@]} --N_FILES ${N_FILES} --PARTIAL_OUTPUT_DIR ${PARTIAL_OUTPUT_DIR} --OUTPUT_DIR ${OUTPUT_DIR} --PREPROCESSING_PIPELINE ${PRE_PIPELINE} --POST_PIPELINE ${POST_PIPELINE} --SUFFIX ${SUFFIX} --NO_FINAL_FILE ${IS_PARTIAL}"
    echo ${CMD}
    preprocess_job_name="preprocess_d${DATASET_LOWER}_${SUFFIX_LOWER}_full_${NOW}"  
    echo "Error: Internal job management tool was to execute the command. Please run the command locally."
fi
