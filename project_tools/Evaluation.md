# Evaluation
------------

We perform two types of evaluation:
* intrinsic
* extrinsic

---

## Intrinsic Evaluation

For intrinsic analysis, consult the notebooks in `/notebooks/Evaluation_Intrinsic`

For per-model evaluation, :
* use `/notebooks/Evaluation_Intrinsic/Evaluate_embeddings_intrinsic.ipynb` for GloVe models
* use `/notebooks/Evaluation_Intrinsic/Evaluation_Intrinsic_BERT_prep.ipynb` to produce static word embeddings from BERT. Then modify the embedding paths in `/notebooks/Evaluation_Intrinsic/Evaluate_embeddings_intrinsic.ipynb` to point to the BERT static embeddings for BERT evaluation

These notebooks will produce summary result output files, which can then be further analyzed for across-model analysis.

For across-model analysis, see:
* `/notebooks/Evaluation_Intrinsic/Evaluate_embeddings - Across Models.ipynb`

We also provide a notebook for a further "deep dive" analysis in:
* `/notebooks/Evaluation_Intrinsic/Evaluate Embeddings - Special Investigation.ipynb`

---

## Extrinsic Evaluation via Keyword Prediction task

The keyword prediction evaluation on the Geology Keyword Prediction dataset is a down-stream evaluation and as such requires us to train models on this task. It also requires us to produce a training and testing dataset.

We formulate this task in a number of ways according to:
* Pairing vs Multiclass
* Text column as Title or Description
* Number of subjects/keywords as 5, 10, or 30

We then split each dataset into 5 (for cross validation) and train multiple runs of each model (with different random seeds) on each fold, taking the final metrics as the average across runs and then across splits.

---

### Producing the keyword prediction dataset

The keyword prediction datasets are created from the Geoscan metadata.

It can be reproduced through the following steps
1) Run the notebook `notebooks/Preprocessing_and_Dataset_Analysis/Metadata analysis.ipynb` to produce a processed form of the metadata file. This processed version is cleaned in order to removed french text etc. It also includes columns with the top 5, top 10, and top 30 subjects/keywords in the dataset, which are used in subsequent steps.
2) Run the notebook `notebooks/Keyword_Prediction_Dataset.ipynb` to produce
    a) datasets for both tasks, description vs title, and varying numbers of subjects (output .parquet files)
    b) splits for each of those datasets created in step a, for cross validation (output .csv for .json files)

For each dataset we produce 5 (default) train/valid splits. We evaluate performance of the model as the average performance across all splits. If any hyperparameter searching needs to be done, it is done by further splitting the training set into train-train/train-valid splits. The goal of this setup is to reduce the effect of picking a single "easy" or "hard" final test dataset.

---

### Training
We can train models for either the MULTICLASS or PAIRING tasks.

The important training parameters for training common to all models are:
* the task (multiclass or pairing)
* the number of subjects (5, 10, 30)
* whether to uset the Title or Description as the input text

There other specific parameters which are specified below.

#### Training the BERT models

Before any finetuning step, the model files need to be put into a separate folder using `scripts/organize_model_folder.py`.

BERT models can be trained on the keyword prediction task via the `run_keyword_prediction_bert()` function in `/nrcan_p2/evaluation/keyword_prediction.py`. We make use of the hf library functionality to train this model. Training is recommended on a GPU.

**NOTE**: Before training, ensure that the provided model and tokenizer paths are accessible to the filesystem on which the code is running. This may require changing the paths in the code. Also, ensure that the model directory contains `config.json` and `pytorch_model.bin`. If you're using an extended tokenizer, make sure that the correct path is set in `tokenizer_config.json`.

To launch training, run one of the following scripts:

For training **locally**:
```
python scripts/run_bert_evaluation.py
    --TASK {MULTICLASS/PAIRING}
    --DATA_DIR /path/to/data/splits
    --USE_CLASS_WEIGHTS {True/False}
    --MODEL_PATH {MODEL_PATH}
```
where task is set to train on either the MULTICLASS or PAIRING keyword prediction tasks. DATA_DIR must point to the training dataset (specifically, the folder that contains the subfolders for each split). USE_CLASS_WEIGHTS is only implemented for MULTICLASS prediction. MODEL_PATH must point to the path of the model or a name of a pretrained model provided by hf.


To use weights and biases (https://wandb.ai) visualizations, set the correct api keys and dir in `scripts/run_bert_evaluation.py` before finetuning. Refer to the wandb documentaion for more details.

#### Training the GloVe models with sklearn
Glove keyword prediction models can be trained with sklearn using the `run_keyword_prediction_classic()` function in `/nrcan_p2/evaluation/keyword_prediction.py`.

To launch training, run one of the following scripts:

For training **locally**:
```
python scripts/run_glove_keyword_evaluation.py
    --TASK {MULTICLASS/PAIRING}
    --DATA_DIR /path/to/data/splits
    --USE_CLASS_WEIGHTS {True/False}
    --EMBEDDING_MODEL_PATH /path/to/embedding/weights.txt
    --CLASSIFICATON_MODEL {RF/MLP}
    --USE_MULTIOUTPUT_WRAPPER {True/False}
```
where task is set to MULTICLASS or PAIRING. DATA_DIR must point to the training dataset (specifically, the folder that contains the subfolders for each split). USE_CLASS_WEIGHTS can be true or false and determines whether or not to run class-rebalanced training. EMBEDDING_MODEL_PATH must point to the embedding model. CLASSIFICATON_MODEL is either RF or MLP to train a Random Forest Classifier or a Multi-layer Perceptron. USE_MULTIOUTPUT_WRAPPER determines whether we train a single "model" with shared weights for each class or an ensemble of models, one for each class.

**NOTE**: Training proceeds using cross validation in order to test out multiple hyperparameter settings for e.g. the RF size and the MLP learning rate. This HP search will be extremely slow unless you set n_jobs > 1 in `run_glove_keyword_evaluation.py`. Furthermore, it will be very slow for the MLP unless you reduce the grid for grid search (defined in `run_glove_keyword_evaluation.py`).


#### Training the GloVe models with Keras

Glove keyword prediction models can be trained with keras (Bi-LSTM models) using the `run_keyword_prediction_keras()` funtion in `/nrcan_p2/evaluation/keyword_prediction.py`.

To launch training, run one of the following scripts:

For training **locally**:
```
python scripts/run_keras_keyword_evaluation.py
    --TASK {MULTICLASS/PAIRING}
    --DATA_DIR /path/to/data/splits
    --USE_CLASS_WEIGHTS {True/False}
    --EMBEDDING_MODEL_PATH /path/to/embedding/weights.txt
    --EXISTING_RUN_DIR /path/to/run/to/finish
```
where task is set to MULTICLASS or PAIRING. DATA_DIR must point to the training dataset (specifically, the folder that contains the subfolders for each split). USE_CLASS_WEIGHTS can be true or false and determines whether or not to run class-rebalanced training. EMBEDDING_MODEL_PATH must point to the embedding model. EXISTING_RUN_DIR should be None for normal training, but can be set to an output folder which contains the partial output of a training job (e.g. if the training crashed and some splits finished while others have not) to restart training for that run where it left off. In this case, all of the other parameters will be ignored and the dataset etc will be loaded from the config in that run's directory.  

```
bash scripts/run_keras_rerun.sh
```
being sure to point to the correct output folder of the partially complete model.

---

### Training outputs

Training outputs results into a timestamped subdirectory in the output directory specified by the training scripts. The outputs differ depending on the model.

#### BERT models
In the subdir:
* input_data.log: text file, listing the input dataset
* split_i_run_j folders for each split i and run j

In the split subfolders:
* log.bin: training log file
* logging: folder, maintains some other log information
* log.txt: lists the number of splits
* model: folder (see below)

In the split folder model subfolder:
* checkpoint-X : various checkpoint folders for the checkpoints saved by the model
* data_args.json: lists the training data arguments passed to DataTrainingArguments() in the training script
* eval_results_None.json: evaluation metrics in json format
* eval_results_None.txt: evaluation metrics in txt format
* label_list.json: json mapping of category IDs to classification category names
* model_args.json: lists the model arguments passed to ModelArguments() in the training script
* pytorch_model.bin: the final model
* special_tokens_map.json: json mapping of special characters to their tokens
* test_results_None.txt: predictions on the test set (only accurate for PAIRING task)
* tokenizer_config.json: config used for the tokenizer initialization
* trainer_state.json: trainer state, including history etc.
* training_args.bin: training arguments passed to TrainingArguments() in the training script (bin version)
* training_args.json: training arguments passed to TrainingArguments() in the training script (bin version)
* train_results.json: training metrics (runtime etc.) (json format)
* trian_results.txt: training metrics (txt format)
* vocab.txt: model vocab

#### GloVe sklearn models
In the subdir:
* input_data.log: json config file that specifies the training parameters, including the input training dataset
* split_i_run_j folders for each split i and run j

In the split subfolders:
* gs.joblib: the output fitted gridsearch object
* metrics.json: the evaluation metrics
* movel_cv_results.json: the results of the gridsearch
* model.joblib: the "best estimator" from the gridsearch (ie the model)

#### GloVe keras models
In the subdir:
* input_data.log: json config file that specifies the training parameters, including the input training dataset
* split_i_run_j folders for each split i and run j

In the split subfolders:
* metrics.json: the evaluation metrics
* model.keras: the trained model
* model.history: the history object (used to restart training, maintains training loss etc.)

---

### Compiling results across models

In order to aggregate the results across the trained models, run one of the following scripts. Note that you must run this on a system that has access to the directories that contain the output from the keyword prediction training step above.

For BERT models:
```
python scripts/get_bert_eval_runs.py
```

For GloVe sklearn models:
```
python scripts/get_glove_eval_runs.py
```

For GloVe keras models:
```
python scripts/get_keras_eval_runs.py
```

---

### Performing analysis

Analysis is performed using the following notebooks. These notebooks require that the model results be compiled via the scripts described above.

For BERT models (MULTICLASS):
* For comparing overall model performance: `/notebooks/Evaluation_Extrinsic/Keyword Prediction Multiclass Evaluation BERT - model comparison.ipynb`
* For comparing performance across subjects: `/notebooks/Evaluation_Extrinsic/Keyword Prediction Multiclass Evaluation BERT - subject comparison and error analysis.ipynb`

For BERT models (PAIRING):
* For comparing overall model performance: `/notebooks/Evaluation_Extrinsic_BERT/Keyword Prediction Pairing Evaluation BERT - model comparison.ipynb`
* For comparing performance across subjects: `/notebooks/Evaluation_Extrinsic_BERT/Keyword Prediction Pairing Evaluation BERT - subject comparison and error analysis.ipynb`

For Glove models:
* For comparing overall model performance: `/notebooks/Evaluation_Extrinsic/Keyword Prediction Evaluation Glove - model comparison.ipynb`
