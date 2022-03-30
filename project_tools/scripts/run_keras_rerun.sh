# Copyright (C) 2021 ServiceNow, Inc.
#
# Rerun an halted and unfinished GloVe keras evaluation job
#

rerun_dirs=(
    #"/nrcan_p2/data/07_model_output/keyword_prediction_keras/run-glove-keras-MULTICLASS_2021-03-08-04-58-56781359"
    "/nrcan_p2/data/07_model_output/keyword_prediction_keras/run-glove-keras-MULTICLASS_2021-03-08-04-54-00019083" # pretrained subj5 v1 False title

)

METHOD="MULTICLASS"
DATA_DIR="None"
EMBEDDING_MODEL_PATH="None"
USE_CLASS_WEIGHTS="n"


for RERUN_DIR in ${rerun_dirs[@]}
do
    NOW=$(date +"%m_%d_%Y_%H_%M_%S")
    METHOD_LOWER=$(echo "${METHOD}" | tr '[:upper:]' '[:lower:]')
    JOB_NAME="keras_evaluation_${METHOD_LOWER}_${NOW}"

    CMD="python scripts/run_keras_keyword_evaluation.py --TASK $METHOD --DATA_DIR $DATA_DIR --EMBEDDING_MODEL_PATH $EMBEDDING_MODEL_PATH --USE_CLASS_WEIGHTS $USE_CLASS_WEIGHTS --EXISTING_RUN_DIR $RERUN_DIR"

    echo ${CMD}
    echo ${JOB_NAME}
    echo "Error: Internal job management tool was to execute the command. Please run the command locally."
done
