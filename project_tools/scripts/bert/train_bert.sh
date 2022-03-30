# Copyright (C) 2021 ServiceNow, Inc.
#
# train_bert.sh
# Trains a DistilBERT model
#
# The full list of parameters and their description can be found at:
# https://github.com/huggingface/transformers/blob/master/src/transformers/training_args.py
# Please see this script for explanations of arguments used to `run_mlm.py` below.

# Create and set output directory
NOW=$(date '+%F_%H%M%S')
OUTPUT_DIR="/nrcan_p2/workspace/lindsay/nrcan_p2/local/data/models_testing/bert_models/model_training_${NOW}"
mkdir $OUTPUT_DIR

# Environment variable used by `run_mlm.py`
export TRANSFORMERS_CACHE='/nrcan_p2/data/local/models_testing/.cache/model_cache'

# Uncomment lines below (and insert appropriate values) to use Weights & Biases for visualization
# export WANDB_API_KEY=<INSERT_WANDB_API_KEY_HERE>
# export WANDB_PROJECT='MyProjectName'

# Model, tokenizer, and data
MODEL_NAME='distilbert-base-uncased'
TOKENIZER='/nrcan_p2/workspace/lindsay/nrcan_p2/local/data/models_testing/tokenizers/bert_geo/bert_geo_250' # distilbert-base-uncased
TRAIN_FILE='/nrcan_p2/workspace/lindsay/nrcan_p2/local/data/data_testing/train_smaller.txt'
VALIDATION_FILE='/nrcan_p2/workspace/lindsay/nrcan_p2/local/data/data_testing/validation_smaller.txt'


LEARNING_RATE=0.00005
WARMUP_STEPS=0
MAX_STEPS=9

echo "Output directory:" $OUTPUT_DIR
echo "Training file:" $TRAIN_FILE
echo "Validation file:" $VALIDATION_FILE
echo "Tokenizer:" $TOKENIZER
echo ""
echo "Model parameters:"
echo "Learning rate:" $LEARNING_RATE
echo "Warmup steps:" $WARMUP_STEPS
echo "Max steps:" $MAX_STEPS


python nrcan_p2/bert_pretraining/run_mlm.py --model_name_or_path ${MODEL_NAME} \
--tokenizer_name ${TOKENIZER} \
--train_file ${TRAIN_FILE} --validation_file ${VALIDATION_FILE} --output_dir ${OUTPUT_DIR} \
--learning_rate ${LEARNING_RATE} --warmup_steps ${WARMUP_STEPS} --max_steps ${MAX_STEPS} \
--max_seq_length 512 --per_device_train_batch_size 12 --per_device_eval_batch_size 12 --line_by_line \
--fp16 \
--do_train --do_eval
