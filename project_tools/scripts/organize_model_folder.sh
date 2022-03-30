# Copyright (C) 2021 ServiceNow, Inc.
#
# Move the model files into a separate folder than trainer_state.json or trainer_args.bin
# Not doing so will cause HuggingFace try to resume training from the state file
#

#!/usr/bin/env bash

DIRS=(
    # model directories that point to config.json
    '/nrcan_p2/mlflow_data/111/aa511d0956044554a6003b03281caa81/model'
    '/nrcan_p2/mlflow_data/122/80c6425f81cf42589bde23cd0a76b25e/model'
)


CMD="python scripts/organize_model_folder.py --DIRS ${DIRS[@]}"

NOW=$(date +"%m_%d_%Y_%H_%M_%S")
JOB_NAME="organize_model_folder_${NOW}"
echo ${CMD}
echo ${JOB_NAME}
echo "Error: Internal job management tool was to execute the command. Please run the command locally."
