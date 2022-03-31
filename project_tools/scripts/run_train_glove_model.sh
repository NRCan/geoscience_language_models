# Copyright (C) 2021 ServiceNow, Inc.
#
# Train a glove model
#
NOW=$(date +"%m_%d_%Y_%H_%M_%S")
JOB_NAME="glove_train_${NOW}"

CMD="bash scripts/train_glove_model.sh" 
echo "Error: Internal job management tool was to execute the command. Please run the command locally." 
