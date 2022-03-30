# Copyright (C) 2021 ServiceNow, Inc.
#
# Train a Mittens (Fine-tuned GloVe) model on a given dataset. 
#
# This assumes training with the NRCan training data
# and default NRCan folder structure. 
# To train on different data and with a different folder structure, 
# simply update the parameters below. 
# 
# outputs the following files to the timestamped model output folder:
#  * config.conf (contains the parameters used in this script)
#  * 
#  * train_glove_model.sh <-- a copy of this script, used to log the input parameters
#  * matrix.npy 
#  * vocab.pkl
#  

# Corpus
DATASET="A_full"
DATA_PIPELINE="SIMPLE_PIPELINE_GLOVE_3_POSTPIPE_GLOVE"
SUFFIX="v1"
INPUT_TEXT="/nrcan_p2/data/03_primary/v4/all_text_${DATA_PIPELINE}_d${DATASET}_${SUFFIX}.txt"

# Original Embeddings
ORIGINAL_EMBEDDINGS_PATH='/nrcan_p2/data/06_models/glove_pretrained/glove.6B.50d.txt'

# Hyperparameters
VOCAB_MIN_COUNT=50
MAX_VOCAB_SIZE=
WINDOW_SIZE=15
VECTOR_SIZE=50
MAX_ITER=100

NOW=$(date +"%m_%d_%Y_%H_%M_%S")

# File names
OUTPUT_FOLDER="/nrcan_p2/data/06_models/mittens/run-${NOW}"

CMD=
if [ ${MAX_VOCAB_SIZE} ]
then
    
CMD="python embeddings/mittens_scripts/run_mittens.py"\
" --OUTPUT_FOLDER ${OUTPUT_FOLDER}"\
" --INPUT_TEXT_FILENAME ${INPUT_TEXT}"\
" --VOCAB_MIN_COUNT ${VOCAB_MIN_COUNT}"\
" --MAX_VOCAB_SIZE ${MAX_VOCAB_SIZE}"\
" --WINDOW_SIZE ${WINDOW_SIZE}"\
" --ORIGINAL_EMBEDDINGS_PATH ${ORIGINAL_EMBEDDINGS_PATH}"\
" --MAX_ITER ${MAX_ITER}"\
" --VECTOR_SIZE ${VECTOR_SIZE}"

else

CMD="python embeddings/mittens_scripts/run_mittens.py"\
" --OUTPUT_FOLDER ${OUTPUT_FOLDER}"\
" --INPUT_TEXT_FILENAME ${INPUT_TEXT}"\
" --VOCAB_MIN_COUNT ${VOCAB_MIN_COUNT}"\
" --WINDOW_SIZE ${WINDOW_SIZE}"\
" --ORIGINAL_EMBEDDINGS_PATH ${ORIGINAL_EMBEDDINGS_PATH}"\
" --MAX_ITER ${MAX_ITER}"\
" --VECTOR_SIZE ${VECTOR_SIZE}"

fi

echo ${CMD}


DATAPIPE_LOWER=$(echo "${DATA_PIPELINE}" | tr '[:upper:]' '[:lower:]')
DATASET_LOWER=$(echo "${DATASET}" | tr '[:upper:]' '[:lower:]')

echo "Error: Internal job management tool was to execute the command. Please run the command locally."
