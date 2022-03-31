# Copyright (C) 2021 ServiceNow, Inc.
#
# Calculate vocab counts for datasets
#
FILES=(
    "/nrcan_p2/data/03_primary/v4/all_text_SIMPLE_PIPELINE_GLOVE_3_POSTPIPE_GLOVE_dA_full_v1.txt"
    "/nrcan_p2/data/03_primary/v4/all_text_PIPELINE_GLOVE_90_POSTPIPE_GLOVE_dA_full_v1.txt"
    "/nrcan_p2/data/03_primary/v4/all_text_PIPELINE_GLOVE_80_POSTPIPE_GLOVE_dA_full_v1.txt"
    "/nrcan_p2/data/03_primary/v4/all_text_PIPELINE_GLOVE_PLUS_POSTPIPE_GLOVE_dA_full_v1.txt"
    "/nrcan_p2/data/03_primary/v4_A_B/all_text_SIMPLE_PIPELINE_GLOVE_3_POSTPIPE_GLOVE_dA_full_dB_v1.txt"
    "/nrcan_p2/data/03_primary/v4_A_B/all_text_PIPELINE_GLOVE_80_POSTPIPE_GLOVE_dA_full_dB_v1.txt"
    "/nrcan_p2/data/03_primary/v4_A_B/all_text_PIPELINE_GLOVE_90_POSTPIPE_GLOVE_dA_full_dB_v1.txt"
    "/nrcan_p2/data/03_primary/v4_A_B/all_text_PIPELINE_GLOVE_PLUS_POSTPIPE_GLOVE_dA_full_dB_v1.txt"
    "/nrcan_p2/data/03_primary/v4_A_B_D/all_text_SIMPLE_PIPELINE_GLOVE_3_POSTPIPE_GLOVE_dA_full_dB_dD_v1.txt"
    "/nrcan_p2/data/03_primary/v4_A_B_D/all_text_PIPELINE_GLOVE_PLUS_POSTPIPE_GLOVE_dA_full_dB_dD_v1.txt"
    "/nrcan_p2/data/03_primary/v4_A_B_D/all_text_PIPELINE_GLOVE_80_POSTPIPE_GLOVE_dA_full_dB_dD_v1.txt"
    "/nrcan_p2/data/03_primary/v4_A_B_D/all_text_PIPELINE_GLOVE_90_POSTPIPE_GLOVE_dA_full_dB_dD_v1.txt"
)

FILES_BERT=(
    "/nrcan_p2/data/03_primary/v4/all_text_SIMPLE_PIPELINE_BERT_3_POSTPIPE_BERT_SPACY_2_dA_full_v1.txt"
    "/nrcan_p2/data/03_primary/v4/all_text_PIPELINE_BERT_90_POSTPIPE_BERT_SPACY_2_dA_full_v1.txt"
    "/nrcan_p2/data/03_primary/v4/all_text_PIPELINE_BERT_80_POSTPIPE_BERT_SPACY_2_dA_full_v1.txt"
    "/nrcan_p2/data/03_primary/v4/all_text_PIPELINE_BERT_PLUS_POSTPIPE_BERT_SPACY_2_dA_full_v1.txt"
    "/nrcan_p2/data/03_primary/v4_A_B/all_text_SIMPLE_PIPELINE_BERT_3_POSTPIPE_BERT_SPACY_2_dA_full_dB_v1.txt"
    "/nrcan_p2/data/03_primary/v4_A_B/all_text_PIPELINE_BERT_80_POSTPIPE_BERT_SPACY_2_dA_full_dB_v1.txt"
    "/nrcan_p2/data/03_primary/v4_A_B/all_text_PIPELINE_BERT_90_POSTPIPE_BERT_SPACY_2_dA_full_dB_v1.txt"
    "/nrcan_p2/data/03_primary/v4_A_B/all_text_PIPELINE_BERT_PLUS_POSTPIPE_BERT_SPACY_2_dA_full_dB_v1.txt"
    "/nrcan_p2/data/03_primary/v4_A_B_D/all_text_SIMPLE_PIPELINE_BERT_3_POSTPIPE_BERT_SPACY_2_dA_full_dB_dD_v1.txt"
    "/nrcan_p2/data/03_primary/v4_A_B_D/all_text_PIPELINE_BERT_PLUS_POSTPIPE_BERT_SPACY_2_dA_full_dB_dD_v1.txt"
    "/nrcan_p2/data/03_primary/v4_A_B_D/all_text_PIPELINE_BERT_90_POSTPIPE_BERT_SPACY_2_dA_full_dB_dD_v1.txt"
    "/nrcan_p2/data/03_primary/v4_A_B_D/all_text_PIPELINE_BERT_80_POSTPIPE_BERT_SPACY_2_dA_full_dB_dD_v1.txt"    
)


for DATA_FILE in ${FILES[@]}
do
    NOW=$(date +"%m_%d_%Y_%H_%M_%S")
    JOB_NAME="vocab_count_glove_${NOW}"

    CMD="python scripts/produce_vocab_counts.py --DATAFILE ${DATA_FILE} --PROCESSING None"
    echo $CMD

    echo "Error: Internal job management tool was to execute the command. Please run the command locally."
done

for DATA_FILE in ${FILES_BERT[@]}
do

    NOW=$(date +"%m_%d_%Y_%H_%M_%S")
    JOB_NAME="vocab_count_bert_none_${NOW}"

    CMD="python scripts/produce_vocab_counts.py --DATAFILE ${DATA_FILE} --PROCESSING None"
    echo $CMD
    echo "Error: Internal job management tool was to execute the command. Please run the command locally."

    NOW=$(date +"%m_%d_%Y_%H_%M_%S")
    JOB_NAME="vocab_count_vert_rm_punct_lower_${NOW}"

    CMD="python scripts/produce_vocab_counts.py --DATAFILE ${DATA_FILE} --PROCESSING rm_punct_lower"
    echo $CMD
    echo "Error: Internal job management tool was to execute the command. Please run the command locally."
done
