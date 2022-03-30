# Copyright (C) 2021 ServiceNow, Inc.
#
# Train a GloVe model on a given dataset. 
#
# This assumes training with the NRCan training data
# and default NRCan folder structure. 
# To train on different data and with a different folder structure, 
# simply update the parameters below. 
# 
# outputs the following files to the timestamped model output folder:
#  * cooccurrence.bin
#  * coocurrence.shuf.bin
#  * train_glove_model.sh <-- a copy of this script, used to log the input parameters
#  * vectors.bin
#  * vectors.txt
#  * vocab.txt

# NRCan training dataset short-form name
DATASET="A_full" # "A_full_dB" "A" "B" "A_full" "A_full_dB_dD"

# Folder in which the dataset file is located
INPUT_FOLDER = "v4" #"v4_B" "v4_A_B" "v4_A_B_D", "v4_D"

NOW=$(date +"%m-%d-%Y-%H-%M-%S")

# NRCan training dataset pipeline
DATA_PIPELINE="SIMPLE_PIPELINE_GLOVE_3_POSTPIPE_GLOVE"

# Suffix associated with the dataset
SUFFIX="v1"

# OUTPUT directory
model_output_dir="/nrcan_p2/data/06_models/glove/dataset_${DATASET}_${NOW}"
input_txt="/nrcan_p2/data/03_primary/${INPUT_FOLDER}/all_text_${DATA_PIPELINE}_d${DATASET}_${SUFFIX}.txt"

# Training parameters
MEMORY=4.0
VOCAB_MIN_COUNT=5
VECTOR_SIZE=250
MAX_ITER=15
WINDOW_SIZE=15
BINARY=2
NUM_THREADS=8
X_MAX=10

echo ${DATASET}
echo ${NOW}
echo ${model_output_dir}
echo ${input_txt}

if [ ! -d ${model_output_dir} ]; then
    mkdir -p ${model_output_dir};
else
    exit 1
fi;

cp scripts/train_glove_model.sh "${model_output_dir}/train_glove_model.sh"

cd embeddings/glove && make

CORPUS=${input_txt} \
    COOCCURRENCE_SHUF_FILE="${model_output_dir}/coocurrence.shuf.bin" \
    COOCCURRENCE_FILE="${model_output_dir}/cooccurrence.bin" \
    SAVE_FILE="${model_output_dir}/vectors" \
    VOCAB_FILE="${model_output_dir}/vocab.txt" \
    MEMORY=${MEMORY} \
    VOCAB_MIN_COUNT=${VOCAB_MIN_COUNT} \
    VECTOR_SIZE=${VECTOR_SIZE} \
    MAX_ITER=${MAX_ITER} \
    WINDOW_SIZE=${WINDOW_SIZE} \
    BINARY=${BINARY} \
    NUM_THREADS=${NUM_THREADS} \
    X_MAX=${X_MAX} \
    bash train.sh 