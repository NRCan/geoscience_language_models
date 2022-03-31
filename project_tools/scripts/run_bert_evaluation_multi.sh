# Copyright (C) 2021 ServiceNow, Inc.
#
# This will batch launch bert keyword prediction evaluation experiments
# for each of the requested parameter settings of
# METHOD, SUBJECT_LIST, TEXT_COL, MODEL_PATHS (and associated PIPELINES)
# USE_CLASS_WEIGHTS_LIST
#

METHOD='PAIRING'

SUBJECT_LIST=(
    'subject_5'
#    'subject_desc_t10'
)

TEXT_COL=(
    'title_merged'
    #'desc_en_en'
#    'desc_en_en_50_3000'
)

MODEL_PATHS=(
    "distilbert-base-uncased"
    #"/nrcan_p2/mlflow_data/42/83d34150748641b89a44b74a1beaf799/model/best_model" #v1 A
    #"/nrcan_p2/mlflow_data/40/902f7aba032a402a927059958a38e079/model/best_model" #v1 A + B
    #"/nrcan_p2/mlflow_data/49/dbc77f67097b4937adcb914e8895ba2c/model/best_model" #v1 A + B + D
    #"/nrcan_p2/mlflow_data/51/9212f6a912664e3d9c949cc35db9d60b/model/best_model" #80 A
    #"/nrcan_p2/mlflow_data/52/16b01e423ebe4f08a0111809eef7830e/model/best_model" #80 A + B
    #"/nrcan_p2/mlflow_data/53/f6d9eefc176241e0b63654c63e97f901/model/best_model" #80 A + B + D
    #"/nrcan_p2/mlflow_data/42/83d34150748641b89a44b74a1beaf799/model/best_model" #v1 A
    #"/nrcan_p2/mlflow_data/40/902f7aba032a402a927059958a38e079/model/best_model" #v1 A + B
    #"/nrcan_p2/mlflow_data/49/dbc77f67097b4937adcb914e8895ba2c/model/best_model" #v1 A + B + D
    #"/nrcan_p2/mlflow_data/51/9212f6a912664e3d9c949cc35db9d60b/model/best_model" #80 A
    #"/nrcan_p2/mlflow_data/52/16b01e423ebe4f08a0111809eef7830e/model/best_model" #80 A + B
    #"/nrcan_p2/mlflow_data/53/f6d9eefc176241e0b63654c63e97f901/model/best_model" #80 A + B + D    
    #"/nrcan_p2/mlflow_data/55/97683cfa22c145faa3e0d1cc64d2d22f/model/best_model" # 90 A + B
    #"/nrcan_p2/mlflow_data/54/9583ad61ff32407bb799716cfe197902/model/best_model" # 90 A + B + D
    #
    #"/nrcan_p2/mlflow_data/58/48c9caa04c6f490d84645e64f8abf99b/model/best_model" # PLUS A 
    #"/nrcan_p2/mlflow_data/59/e99fa92098fe4d99b990523245d7c88c/model/best_model" # PLUS A + B 
    # "/nrcan_p2/mlflow_data/60/47dd5bbf53fd41a19af10114b040f73f/model/best_model" # PLUS A + B + D 
    # "/nrcan_p2/mlflow_data/111/aa511d0956044554a6003b03281caa81/model/best_model" # v1 A geo250 
    #"/nrcan_p2/mlflow_data/122/80c6425f81cf42589bde23cd0a76b25e/model/best_model" # v1 A + B geo250 
    #"/nrcan_p2/mlflow_data/117/cf48bd9e386c4016b91f2d4a6de105fb/model/best_model" # v1 A + B + D geo250 
    #"/nrcan_p2/mlflow_data/115/c9ed23ca6b85429aacaa4d4430383066/model/best_model" # 80 A geo250 
    #"/nrcan_p2/mlflow_data/116/4bb189b9348042d09d6739d55acd70d8/model/best_model" # 80 A + B geo250 #rerun 5
    #"/nrcan_p2/mlflow_data/123/2a668c76d62e408f98c2e20e39ffe837/model/best_model" # 80 A + B + D geo250 #rerun 3 #TODO
    # # OVERFIT # 90 A geo250
    #"/nrcan_p2/mlflow_data/121/ef9952d289794ba78b6b597c1871069c/model/best_model" # 90 A + B geo250 #rerun 2
    # "/nrcan_p2/mlflow_data/118/b09e1d46195e457d8113c13828522fdb/model/best_model" # 90 A + B + D geo250 #rerun 5 ????
    # "/nrcan_p2/mlflow_data/112/23646f5808e648fca301bdab453c9af0/model/best_model" # PLUS A geo250 
    # "/nrcan_p2/mlflow_data/114/a31bd6a676e34f0ab168a09f077ec93f/model/best_model" # PLUS A + B geo250 
    # "/nrcan_p2/mlflow_data/124/6a053efc28174894bb39f2a6a7e10ddf/model/best_model" # PLUS A + B + D geo250 
    #"/nrcan_p2/mlflow_data/96/9b369d7fb505460589f89f58a7d16bde/model/best_model" # v1 A geo500 #rerun 4
    #"/nrcan_p2/mlflow_data/103/fd5ab1b5d832460395215f71fc024333/model/best_model" # v1 A + B + D geo500 #rerun 3
    #"/nrcan_p2/mlflow_data/98/c9d7feab3c3143d2b84c88de0119dca3/model/best_model" # 80 A geo500 # rerun 5
    # "/nrcan_p2/mlflow_data/102/f4bf6a9b7013428c89140d39a3bf2c59/model/best_model" # 80 A + B geo 500 # rerun 3
    #"/nrcan_p2/mlflow_data/106/3258f9f79451477f87a639446295fba0/model/best_model" # 80 A + B + D geo 500 # rerun 2
    # "/nrcan_p2/mlflow_data/104/dd946c63515e460a801e0631487cb1a5/model/best_model" # 90 A + B geo 500 # rerun 3
    # "/nrcan_p2/mlflow_data/107/783b4970e8104255a8fed894b37ddc7a/model/best_model" # 90 A + B + D geo 500 #rerun 1
    # "/nrcan_p2/mlflow_data/97/328ae3b4fa2a4b3ab94940d75422992f/model/best_model" # PLUS A geo500 # rerun 3
    # "/nrcan_p2/mlflow_data/101/6de217a53bac450fa2e63fc2420a3d73/model/best_model" # PLUS A + B geo500 # rerun 2
    # "/nrcan_p2/mlflow_data/105/9ea9830e8e4c4bb3be386be4605e5a68/model/best_model" # PLUS A + B + D geo500 #rerun 3
    #"/nrcan_p2/mlflow_data/119/58541a9131dc40248198702d06e08d34/model/best_model" # A + B geo250 3m #rerun 6
    #"/nrcan_p2/mlflow_data/120/ac505d9ae17e4a648fb15224dbb0ab10/model/best_model" # A + B geo250 3m #TODO
#    "/nrcan_p2/mlflow_data/81/108721072d4840e78509f27ed2554422/model/best_model" # 994
)

# the list of pipelines must match the list of models in length
PIPELINES=(
    #"None_None"
    #"SIMPLE_PIPELINE_BERT_3_None"
    #"SIMPLE_PIPELINE_BERT_3_None"
    #"SIMPLE_PIPELINE_BERT_3_None"
    #"PIPELINE_BERT_80_None"
    #"PIPELINE_BERT_80_None"
    #"PIPELINE_BERT_80_None"
    # "None_None"
    # "None_None"
    # "None_None"
    # "None_None"
    # "None_None"
    # "None_None"
    # "None_None"
    # "None_None"
    # "None_None"
    "None_None"
    # "None_None"
    # "None_None"
    # "None_None"
    #"None_None"
    #"None_None"
    #"None_None"
    #"None_None"                
    #"PIPELINE_BERT_90_None"
    #"PIPELINE_BERT_PLUS_None"
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
        #"False"
        "True"
    )
else
echo "Unknown METHOD {$METHOD}"
exit 1
fi

#make push

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
                NOW=$(date +"%m_%d_%Y_%H_%M_%S")
                METHOD_LOWER=$(echo "${METHOD}" | tr '[:upper:]' '[:lower:]')
                JOB_NAME="bert_evaluation_${METHOD_LOWER}_${NOW}"

                echo " "
                echo "METHOD: ${METHOD}"
                echo "DATA_DIR: ${DATA_DIR}"
                echo "MODEL_PATH: ${MODEL_PATH}"
                CMD="python scripts/run_bert_evaluation.py --TASK $METHOD --DATA_DIR $DATA_DIR --MODEL_PATH $MODEL_PATH --USE_CLASS_WEIGHTS $USE_CLASS_WEIGHTS"

                echo ${CMD}
                echo ${JOB_NAME}
		echo "Error: Internal job management tool was to execute the command. Please run the command locally."
            done
        done
    done
done
