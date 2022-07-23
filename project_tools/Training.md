# Training Embeddings Models

You can train 3 types of embedddings models:
* GloVe vectors
* Mittens vectors
* BERT vectors (through continued pretraining)

---

## GloVe

### Local training
To train a GloVe model, run the following command after setting the correct dataset and training parameters in the script (see the script for more details):
```
bash scripts/train_glove_model.sh
```

At minimum, you will probably need to change the following parameters:
* DATASET
* INPUT_FOLDER
* DATA_PIPELINE
* model_output_dir
* input_txt
* VECTOR_SIZE

This will first compile (via `make`) the GloVe code in `embeddings/glove` and launch `embeddings/glove/train.sh`.  

---

## Mittens

### Local training
To train a Mittens model, run the following python script with the appropriate input parameters. See the script for more details:
```
python embeddings/mittens_scripts/run_mittens.py
    --OUTPUT_FOLDER ${OUTPUT_FOLDER}
    --INPUT_TEXT_FILENAME ${INPUT_TEXT}
    --VOCAB_MIN_COUNT ${VOCAB_MIN_COUNT}
    --WINDOW_SIZE ${WINDOW_SIZE}
    --ORIGINAL_EMBEDDINGS_PATH ${ORIGINAL_EMBEDDINGS_PATH}
    --MAX_ITER ${MAX_ITER}
    --VECTOR_SIZE ${VECTOR_SIZE}
```

At minimum, you will probably need to change the following parameters:
* DATASET
* DATA_PIPELINE
* INPUT_TEXT
* ORIGINAL_EMBEDDINGS_PATH
* VECTOR_SIZE

---

## BERT

### Train/Validation split

The Dataset needs to be split into training and validation sets. Then the validation set can be used to check if the model is learning (i.e. the validation loss is dropping).

To do the split, run:

```
python split_data.py --INPUT_FILE path/to/input/file.txt --OUTPUT_DIR path/to/output/directory

```
which splits `INPUT_FILE` into `train.txt` and `validation.txt` and saves them to `OUTPUT_DIR`.

Note that `INPUT_FILE` needs to be a `txt` file with one sentence on each line.

### Further pretraining

Pretrained BERT models were further pretrained on Geology-specific data. The below scripts run further pretraining of a BERT model, specifically `Distilbert-base-uncased` (but could be modified to run other transformer models in the Huggingface repository), with a pretrained or custom tokenizer.

Prerequisites:
* Enough GPU memory (or change your batch size accordingly)
* Enough CPU memory (64 GB suggested; less can work for small enough datasets)
* Training and validation datasets
* A [Weights and Biases](https://www.wandb.ai) account and API key (optional; required for visualization with W&B)

To run training, modify `nrcan_p2/scripts/bert/train_bert.sh` as appropriate (see below) and then run from the `nrcan_p2` project directory, i.e.:

```
source scripts/bert/train_bert.sh
```

This will run `nrcan_p2/nrcan_p2/bert_pretraining/run_mlm.py` with the experimental parameters, as well as hyperparameters that can be adjusted.

###### Key variables to set in `train_bert.sh`:

Models and input:

***Note*** *that the paths below will have to be changed appropriately.*

- `MODEL_NAME`: `distilbert-base-uncased` has been tested, but one could test other models in the Huggingface repository
- `TOKENIZER`:
  - For a custom tokenizer, point to its directory (e.g., `'/nrcan_p2/workspace/lindsay/nrcan_p2/local/data/models_testing/tokenizers/bert_geo/bert_geo_250'`).
  - For a pretrained tokenizer, use the model name (e.g., `distilbert-base-uncased`)
- `TRAIN_FILE`: Full path to your training dataset
* `VALIDATION_FILE`: Full path to your validation dataset

Model hyperparameters (see full list [here](https://github.com/huggingface/transformers/blob/master/src/transformers/training_args.py)):

*Variables in all caps are set at the top of the script for ease of use, but others can be set as variables passed to `run_mlm.py` directly.*
- `LEARNING_RATE` (passed to `run_mlm.py` as `learning_rate`): Initial learning rate for the Adam optimizer. In our experiments, 0.00005 often had good performance, but others may work better with other combinations of hyperparameters
- `WARMUP_STEPS` (passed to `run_mlm.py` as `warmup_steps`): Number of steps used for a linear warmup from 0 to `learning_rate`
- `MAX_STEPS` (passed to `run_mlm.py` as `max_steps`): The total number of training steps to perform
- You may want to experiment with `per_device_train_batch_size` and `per_device_eval_batch_size` if you're running out of memory.
- `--line_by_line` and `--max_seq_length 512` were used for all DistilBERT training experiments

---
