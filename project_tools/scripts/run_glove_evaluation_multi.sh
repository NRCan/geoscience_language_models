# Copyright (C) 2021 ServiceNow, Inc.
#
# This will batch launch GloVe keyword prediction evaluation experiments
# for each of the requested parameter settings of
# METHOD, SUBJECT_LIST, TEXT_COL, MODEL_PATHS (and associated PIPELINES)
# CLASSIFIER_LIST, USE_MULTIOUTPUT_WRAPPER_LIST, USE_CLASS_WEIGHTS_LIST
#

METHOD='MULTICLASS'

SUBJECT_LIST=(
    #'subject_5'
    'subject_30'
    'subject_desc_t10'
)

TEXT_COL=(
    'title_merged'
    #'desc_en_en'
    'desc_en_en_50_3000'
)

MODEL_PATHS=(  
    #"/nrcan_p2/data/06_models/glove_pretrained/glove.6B.50d.txt"
    #"/nrcan_p2/data/06_models/glove/dataset_A_full_02-01-2021-03-52-42/vectors.txt" #v1 A
    # "/nrcan_p2/data/06_models/glove/dataset_A_full_dB_02-01-2021-03-49-47/vectors.txt" #v1 A + B
    # "/nrcan_p2/data/06_models/glove/dataset_A_full_dB_dD_02-16-2021-18-12-27/vectors.txt" #v1 A + B + D
    # "/nrcan_p2/data/06_models/glove/dataset_A_full_02-23-2021-02-46-38" #80 A
    # "/nrcan_p2/data/06_models/glove/dataset_A_full_dB_02-23-2021-15-33-49" #80 A + B
    # "/nrcan_p2/data/06_models/glove/dataset_A_full_dB_dD_02-23-2021-02-56-14" #80 A + B + D
    # "/nrcan_p2/data/06_models/glove/dataset_A_full_02-23-2021-02-47-53/vectors.txt" # 90 A
    # "/nrcan_p2/data/06_models/glove/dataset_A_full_dB_02-23-2021-02-43-30/vectors.txt" # 90 A + B
    # "/nrcan_p2/data/06_models/glove/dataset_A_full_dB_dD_02-23-2021-02-54-08/vectors.txt" # 90 A + B + D
    # "/nrcan_p2/data/06_models/glove/dataset_A_full_02-23-2021-02-49-19/vectors.txt" # PLUS A
    # "/nrcan_p2/data/06_models/glove/dataset_A_full_dB_02-23-2021-15-36-31/vectors.txt" # PLUS A + B
    # "/nrcan_p2/data/06_models/glove/dataset_A_full_dB_dD_02-23-2021-02-52-50/vectors.txt" # PLUS A + B + D
    # "/nrcan_p2/data/06_models/glove_pretrained/glove.6B.300d.txt" # 300d
    "/nrcan_p2/data/06_models/glove/dataset_A_full_02-24-2021-23-00-02/vectors.txt" # v1 A 300d
    "/nrcan_p2/data/06_models/glove/dataset_A_full_dB_02-28-2021-19-49-35/vectors.txt" # v1 A + B 300d <- subj 30 t10 and all mlp
    "/nrcan_p2/data/06_models/glove/dataset_A_full_dB_dD_02-28-2021-19-50-45/vectors.txt" # v1 A + B + D 300d <-- subj 30, t10 and all mlp
    # "/nrcan_p2/data/06_models/glove/dataset_A_full_02-24-2021-18-56-28/vectors.txt" # 80 A 300d
    # "/nrcan_p2/data/06_models/glove/dataset_A_full_dB_02-28-2021-19-57-01/vectors.txt" # 80 A + B 300d
    # "/nrcan_p2/data/06_models/glove/dataset_A_full_dB_dD_02-28-2021-19-54-45/vectors.txt" # 80 A + B + D 300d
    # "/nrcan_p2/data/06_models/glove/dataset_A_full_02-24-2021-23-19-15/vectors.txt" # 90 A 300d
    # "/nrcan_p2/data/06_models/glove/dataset_A_full_dB_02-28-2021-20-07-54/vectors.txt" # 90 A + B 300d
    # "/nrcan_p2/data/06_models/glove/dataset_A_full_dB_dD_02-28-2021-20-08-52/vectors.txt" # 90 A + B + D 300d
    # "/nrcan_p2/data/06_models/glove/dataset_A_full_02-24-2021-18-40-40/vectors.txt" # PLUS A 300d
    # "/nrcan_p2/data/06_models/glove/dataset_A_full_dB_02-28-2021-20-15-12/vectors.txt" # PLUS A + B 300d
    # "/nrcan_p2/data/06_models/glove/dataset_A_full_dB_dD_02-25-2021-01-56-02/vectors.txt" # PLUS A + B + D 300d
)
PIPELINES=(
    #"SIMPLE_PIPELINE_GLOVE_3_POSTPIPE_GLOVE"
    # "SIMPLE_PIPELINE_GLOVE_3_POSTPIPE_GLOVE"
    # "SIMPLE_PIPELINE_GLOVE_3_POSTPIPE_GLOVE"
    # "SIMPLE_PIPELINE_GLOVE_3_POSTPIPE_GLOVE"
    # "PIPELINE_GLOVE_80_None"
    # "PIPELINE_GLOVE_80_None"
    # "PIPELINE_GLOVE_80_None"
    # "PIPELINE_GLOVE_90_None"
    # "PIPELINE_GLOVE_90_None"
    # "PIPELINE_GLOVE_90_None"    
    # "PIPELINE_GLOVE_PLUS_None"
    # "PIPELINE_GLOVE_PLUS_None"
    # "PIPELINE_GLOVE_PLUS_None"    
    #"SIMPLE_PIPELINE_GLOVE_3_POSTPIPE_GLOVE"
    "SIMPLE_PIPELINE_GLOVE_3_POSTPIPE_GLOVE"
    "SIMPLE_PIPELINE_GLOVE_3_POSTPIPE_GLOVE"
    "SIMPLE_PIPELINE_GLOVE_3_POSTPIPE_GLOVE"
    # "PIPELINE_GLOVE_80_POSTPIPE_GLOVE"
    # "PIPELINE_GLOVE_80_POSTPIPE_GLOVE"
    # "PIPELINE_GLOVE_80_POSTPIPE_GLOVE"
    # "PIPELINE_GLOVE_90_POSTPIPE_GLOVE"
    # "PIPELINE_GLOVE_90_POSTPIPE_GLOVE"
    # "PIPELINE_GLOVE_90_POSTPIPE_GLOVE"    
    # "PIPELINE_GLOVE_PLUS_POSTPIPE_GLOVE"
    # "PIPELINE_GLOVE_PLUS_POSTPIPE_GLOVE"
    # "PIPELINE_GLOVE_PLUS_POSTPIPE_GLOVE"        
)

CLASSIFIER_LIST=(
    "RF"
    #"MLP"
)

USE_MULTIOUTPUT_WRAPPER_LIST=(
    "False"
    "True"
)

USE_CLASS_WEIGHTS_LIST=
if [ "$METHOD" == 'PAIRING' ] 
then
    USE_CLASS_WEIGHTS_LIST=(
        "False"
    )
elif [ "$METHOD" == 'MULTICLASS' ] 
then
    USE_CLASS_WEIGHTS_LIST=(
        "False"
        "True"
    )
else
echo "Unknown METHOD {$METHOD}"
exit 1
fi


for subj in "${SUBJECT_LIST[@]}"
do
    for text_col in "${TEXT_COL[@]}"
    do
        COUNT=${#MODEL_PATHS[@]}
        for ((i=0; i<$COUNT; i++))
        do
            pipeline=${PIPELINES[i]}
            MODEL_PATH=${MODEL_PATHS[i]}

            DATA_DIR=
            if [ "$METHOD" == 'PAIRING' ] 
            then
            DATA_DIR="/nrcan_p2/data/03_primary/keyword_prediction/splits/PAIRING_small_${subj}_${text_col}_${pipeline}_nNone-Feb29"
            elif [ "$METHOD" == 'MULTICLASS' ] 
            then
            DATA_DIR="/nrcan_p2/data/03_primary/keyword_prediction/splits/MULTICLASS_small_${subj}_${text_col}_${pipeline}_nodrop-Feb29"
            else
            echo "Unknown METHOD {$METHOD}"
            exit 1
            fi

            for USE_CLASS_WEIGHTS in ${USE_CLASS_WEIGHTS_LIST[@]}
            do
                for CLASSIFIER in ${CLASSIFIER_LIST[@]}
                do
                    for USE_MULTIOUTPUT_WRAPPER in ${USE_MULTIOUTPUT_WRAPPER_LIST[@]}
                    do
                        NOW=$(date +"%m_%d_%Y_%H_%M_%S")
                        METHOD_LOWER=$(echo "${METHOD}" | tr '[:upper:]' '[:lower:]')
                        JOB_NAME="glove_evaluation4_${METHOD_LOWER}_${NOW}"

                        echo " "
                        echo "METHOD: ${METHOD}"
                        echo "DATA_DIR: ${DATA_DIR}"
                        echo "MODEL_PATH: ${MODEL_PATH}"
                        CMD="python scripts/run_glove_keyword_evaluation.py --TASK $METHOD --DATA_DIR $DATA_DIR --EMBEDDING_MODEL_PATH $MODEL_PATH --USE_CLASS_WEIGHTS $USE_CLASS_WEIGHTS --CLASSIFICATION_MODEL $CLASSIFIER --USE_MULTIOUTPUT_WRAPPER $USE_MULTIOUTPUT_WRAPPER"

                        echo ${CMD}
                        echo ${JOB_NAME}
			echo "Error: Internal job management tool was to execute the command. Please run the command locally."
                    done
                done
            done
        done
    done
done
