# coding=utf-8
# Copyright (C) 2021 ServiceNow, Inc.
""" Finetuning for running PAIRING and MULTICLASS-MULTILABEL classification finetuning
    on NRCan data and the original Glue benchmarks

    This script is a SIGNIFICANTLY MODIFIED version of the run_glue.py script provided by 
    HuggingFace in the Transformers repo:
        https://github.com/huggingface/transformers/blob/master/examples/text-classification/run_glue.py 
    It has been extended to perform multiclass sequence classification and sentence pairing 
    classification and to handle the format of our inputs.

    It was originally make accessible by hf using the apache 2.0 licence. 
"""

""" Finetuning the library models for sequence classification on GLUE."""
# You can also adapt this script on your own text classification task. Pointers for this are left as comments.

import logging
import os
import numpy as np
import torch
import json
import shutil
import random
import sys
from dataclasses import dataclass, field
from typing import Optional
import pandas as pd

# HACK: hf datasets repo cannot detect disk space on a cluster,
# so let it know there's some space left
shutil.disk_usage = lambda x: shutil._ntuple_diskusage(1, 1, 1)

from datasets import load_dataset, load_metric

from sklearn.metrics import (
    accuracy_score, 
    precision_recall_fscore_support,
    multilabel_confusion_matrix,
    confusion_matrix
)

import transformers
from transformers import (
    BertTokenizer,
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    EvalPrediction,
    HfArgumentParser,
    PretrainedConfig,
    Trainer,
    TrainingArguments,
    default_data_collator,
    set_seed,
)
from transformers.trainer_utils import is_main_process
from nrcan_p2.evaluation.hf_modeling_auto import AutoModelForMultiLabelSequenceClassification


task_to_keys = {
    "cola": ("sentence", None),
    "mnli": ("premise", "hypothesis"),
    "mrpc": ("sentence1", "sentence2"),
    "qnli": ("question", "sentence"),
    "qqp": ("question1", "question2"),
    "rte": ("sentence1", "sentence2"),
    "sst2": ("sentence", None),
    "stsb": ("sentence1", "sentence2"),
    "wnli": ("sentence1", "sentence2"),
}

logger = logging.getLogger(__name__)


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    Using `HfArgumentParser` we can turn this class
    into argparse arguments to be able to specify them on
    the command line.
    """
    label_column_name: str = field(metadata={"help": "Which column contains the label"})

    task_name: Optional[str] = field(
        default=None,
        metadata={"help": "The name of the task to train on: " + ", ".join(task_to_keys.keys())},
    )
    max_seq_length: int = field(
        default=128,
        metadata={
            "help": "The maximum total input sequence length after tokenization. Sequences longer "
            "than this will be truncated, sequences shorter will be padded."
        },
    )
    dataset_cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where do you want to store the datasets models downloaded from huggingface.co"},
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached preprocessed datasets or not."}
    )
    pad_to_max_length: bool = field(
        default=True,
        metadata={
            "help": "Whether to pad all samples to `max_seq_length`. "
            "If False, will pad the samples dynamically when batching to the maximum length in the batch."
        },
    )
    train_file: Optional[str] = field(
        default=None, metadata={"help": "A csv or a json file containing the training data."}
    )
    validation_file: Optional[str] = field(
        default=None, metadata={"help": "A csv or a json file containing the validation data."}
    )
    test_file: Optional[str] = field(default=None, metadata={"help": "A csv or a json file containing the test data."})

    def __post_init__(self):
        if self.task_name is not None:
            self.task_name = self.task_name.lower()
            if self.task_name not in task_to_keys.keys():
                raise ValueError("Unknown task, you should pick one in " + ",".join(task_to_keys.keys()))
        elif self.train_file is None or self.validation_file is None:
            raise ValueError("Need either a GLUE task or a training/validation file.")
        else:
            train_extension = self.train_file.split(".")[-1]
            assert train_extension in ["csv", "json"], "`train_file` should be a csv or a json file."
            validation_extension = self.validation_file.split(".")[-1]
            assert (
                validation_extension == train_extension
            ), "`validation_file` should have the same extension (csv or json) as `train_file`."


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """
    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where do you want to store the pretrained models downloaded from huggingface.co"},
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
    )
    model_revision: str = field(
        default="main",
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )
    use_auth_token: bool = field(
        default=False,
        metadata={
            "help": "Will use the token generated when running `transformers-cli login` (necessary to use this script "
            "with private models)."
        },
    )
    use_class_weights: bool = field(
        default=False, metadata={"help": "Use weights for each class in a multiclass or pairing setting"}
    )


def main():
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.
    shutil.disk_usage = lambda x: shutil._ntuple_diskusage(1, 1, 1)

    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    if (
        os.path.exists(training_args.output_dir)
        and os.listdir(training_args.output_dir)
        and training_args.do_train
        and not training_args.overwrite_output_dir
    ):
        raise ValueError(
            f"Output directory ({training_args.output_dir}) already exists and is not empty. "
            "Use --overwrite_output_dir to overcome."
        )

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if is_main_process(training_args.local_rank) else logging.WARN,
    )

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    # Set the verbosity to info of the Transformers logger (on main process only):
    if is_main_process(training_args.local_rank):
        transformers.utils.logging.set_verbosity_info()
        transformers.utils.logging.enable_default_handler()
        transformers.utils.logging.enable_explicit_format()
    logger.info(f"Training/evaluation parameters {training_args}")

    # Set seed before initializing model.
    set_seed(training_args.seed)

    # Get the datasets: you can either provide your own CSV/JSON training and evaluation files (see below)
    # or specify a GLUE benchmark task (the dataset will be downloaded automatically from the datasets Hub).
    #
    # For CSV/JSON files, this script will use as labels the column called 'label' and as pair of sentences the
    # sentences in columns called 'sentence1' and 'sentence2' if such column exists or the first two columns not named
    # label if at least two columns are provided.
    #
    # If the CSVs/JSONs contain only one non-label column, the script does single sentence classification on this
    # single column. You can easily tweak this behavior (see below)
    #
    # In distributed training, the load_dataset function guarantee that only one local process can concurrently
    # download the dataset.
    label_column_name = data_args.label_column_name
    if data_args.task_name is not None:
        # Downloading and loading a dataset from the hub.
        datasets = load_dataset("glue", data_args.task_name)
    else:
        # Loading a dataset from your local files.
        # CSV/JSON training and evaluation files are needed.
        data_files = {"train": data_args.train_file, "validation": data_args.validation_file}

        # Get the test dataset: you can provide your own CSV/JSON test file (see below)
        # when you use `do_predict` without specifying a GLUE benchmark task.
        if training_args.do_predict:
            if data_args.test_file is not None:
                train_extension = data_args.train_file.split(".")[-1]
                test_extension = data_args.test_file.split(".")[-1]
                assert (
                    test_extension == train_extension
                ), "`test_file` should have the same extension (csv or json) as `train_file`."
                data_files["test"] = data_args.test_file
            else:
                raise ValueError("Need either a GLUE task or a test file for `do_predict`.")

        for key in data_files.keys():
            logger.info(f"load a local file for {key}: {data_files[key]}")

        if data_args.train_file.endswith(".csv"):
            # Loading a dataset from local csv files
            datasets = load_dataset("csv", data_files=data_files, cache_dir=data_args.dataset_cache_dir)
        else:
            # Loading a dataset from local json files
            datasets = load_dataset("json", data_files=data_files, field='data', cache_dir=data_args.dataset_cache_dir)

    # See more about loading any type of standard or custom dataset at
    # https://huggingface.co/docs/datasets/loading_datasets.html.

    # Labels
    if data_args.task_name is not None:
        is_regression = data_args.task_name == "stsb"
        is_multilabel = None
        if not is_regression:
            label_list = datasets["train"].features[label_column_name].names
            num_labels = len(label_list)
        else:
            num_labels = 1
    else:
        # Trying to have good defaults here, don't hesitate to tweak to your needs.
        print(list(datasets['train'].features.keys())[0:10])
        
        is_regression = datasets["train"].features[label_column_name].dtype in ["float32", "float64"]
        # if is_regression:
        #     num_labels = 1
        # else:
        is_multilabel = datasets["train"].features[label_column_name].dtype in [list, "list"]

        # A useful fast method:
        # https://huggingface.co/docs/datasets/package_reference/main_classes.html#datasets.Dataset.unique
        if is_multilabel:
            logger.warning("Found list type label, converting to pandas. This may be slow and memory intensive")
            # this is inefficient
            label_list = datasets["train"].data[label_column_name].to_pandas().explode().unique()
            label_list = [x for x in label_list if not pd.isnull(x)]
            print(len(label_list))
            print(label_list[0:10])
            label_list.sort()  # Let's sort it for determinism
            num_labels = len(label_list)
        else:
            if is_regression:
                num_labels = 1
            else:
                label_list = datasets["train"].unique(label_column_name)
                label_list.sort()  # Let's sort it for determinism
                num_labels = len(label_list)

    # Load pretrained model and tokenizer
    #
    # In distributed training, the .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.
    config = AutoConfig.from_pretrained(
        model_args.config_name if model_args.config_name else model_args.model_name_or_path,
        num_labels=num_labels,
        finetuning_task=data_args.task_name,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )

    # Load a BertTokenizer directly if we have passed it a tokenizer path, 
    # otherwise, use the AutoTokenizer
    if os.path.isdir(model_args.model_name_or_path):
        tokenizer = BertTokenizer.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=model_args.cache_dir,
            use_fast=model_args.use_fast_tokenizer,
            revision=model_args.model_revision,
            use_auth_token=True if model_args.use_auth_token else None,
        )
    else:
        tokenizer = AutoTokenizer.from_pretrained(
            model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
            cache_dir=model_args.cache_dir,
            use_fast=model_args.use_fast_tokenizer,
            revision=model_args.model_revision,
            use_auth_token=True if model_args.use_auth_token else None,
        )    

    if not is_multilabel:
        logging.info("Instantiating multiclass (not multilabel) classification model...")
        model = AutoModelForSequenceClassification.from_pretrained(
            model_args.model_name_or_path,
            from_tf=bool(".ckpt" in model_args.model_name_or_path),
            config=config,
            cache_dir=model_args.cache_dir,
            revision=model_args.model_revision,
            use_auth_token=True if model_args.use_auth_token else None,
        )
    else:
        logging.info("Instantiating multilabel classification model...")
        model = AutoModelForMultiLabelSequenceClassification.from_pretrained(
            model_args.model_name_or_path,
            from_tf=bool(".ckpt" in model_args.model_name_or_path),
            config=config,
            cache_dir=model_args.cache_dir,
            revision=model_args.model_revision,
            use_auth_token=True if model_args.use_auth_token else None,
        )        

    # Preprocessing the datasets
    if data_args.task_name is not None:
        sentence1_key, sentence2_key = task_to_keys[data_args.task_name]
    else:
        # Again, we try to have some nice defaults but don't hesitate to tweak to your use case.
        non_label_column_names = [name for name in datasets["train"].column_names if name != label_column_name]
        print(non_label_column_names)
        if "sentence1" in non_label_column_names and "sentence2" in non_label_column_names:
            sentence1_key, sentence2_key = "sentence1", "sentence2"
        else:
            if len(non_label_column_names) >= 2:
                sentence1_key, sentence2_key = non_label_column_names[:2]
                print(sentence1_key, sentence2_key)
            else:
                sentence1_key, sentence2_key = non_label_column_names[0], None

    # Padding strategy
    if data_args.pad_to_max_length:
        padding = "max_length"
    else:
        # We will pad later, dynamically at batch creation, to the max sequence length in each batch
        padding = False

    # Some models have set the order of the labels to use, so let's make sure we do use it.
    label_to_id = None
    if (
        model.config.label2id != PretrainedConfig(num_labels=num_labels).label2id
        and data_args.task_name is not None
        and is_regression
    ):
        # Some have all caps in their config, some don't.
        label_name_to_id = {k.lower(): v for k, v in model.config.label2id.items()}
        if list(sorted(label_name_to_id.keys())) == list(sorted(label_list)):
            label_to_id = {i: label_name_to_id[label_list[i]] for i in range(num_labels)}
        else:
            logger.warn(
                "Your model seems to have been trained with labels, but they don't match the dataset: ",
                f"model labels: {list(sorted(label_name_to_id.keys()))}, dataset labels: {list(sorted(label_list))}."
                "\nIgnoring the model labels as a result.",
            )
    elif data_args.task_name is None and not is_regression:
        label_to_id = {v: i for i, v in enumerate(label_list)}

    def preprocess_function(examples):
        # Tokenize the texts
        # Replace all null values with "" (these may exist in the validation set)
        args = (
            ([s if s is not None else "" for s in examples[sentence1_key]],
            ) if sentence2_key is None 
            else ([s if s is not None else "" for s in examples[sentence1_key]], 
                [s if s is not None else "" for s in examples[sentence2_key]])
        )
        result = tokenizer(*args, padding=padding, max_length=data_args.max_seq_length, truncation=True)

        # Map labels to IDs (not necessary for GLUE tasks)
        if label_to_id is not None and label_column_name in examples:
            if type(examples[label_column_name][0]) == list:
                label_keys_sorted = sorted(label_to_id.keys())
                result[label_column_name] = [tuple([1 if key in l else 0 for key in label_keys_sorted]) for l in examples[label_column_name]]                
            else:
                result[label_column_name] = [label_to_id[l] for l in examples[label_column_name]]

        return result

    datasets = datasets.map(preprocess_function, batched=True, load_from_cache_file=not data_args.overwrite_cache)

    train_dataset = datasets["train"]
    eval_dataset = datasets["validation_matched" if data_args.task_name == "mnli" else "validation"]
    if data_args.task_name is not None or data_args.test_file is not None:
        test_dataset = datasets["test_matched" if data_args.task_name == "mnli" else "test"]

    # Log a few random samples from the training set:
    for index in random.sample(range(len(train_dataset)), 3):
        logger.info(f"Sample {index} of the training set: {train_dataset[index]}.")

    # Get the metric function
    if data_args.task_name is not None:
        metric = load_metric("glue", data_args.task_name)

    if model_args.use_class_weights:
        if is_multilabel:
            label_np = datasets["train"].data["label"].to_numpy()
            pos_weights = label_np.shape[0]/np.stack(label_np).sum(axis=0)
            model.pos_weights = torch.tensor(pos_weights, device=torch.device("cuda") if torch.cuda.is_available() else "cpu")
    # TODO: When datasets metrics include regular accuracy, make an else here and remove special branch from
    # compute_metrics

    # You can define your custom compute_metrics function. It takes an `EvalPrediction` object (a namedtuple with a
    # predictions and label_ids field) and has to return a dictionary string to float.
    def compute_metrics(p: EvalPrediction):
        preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
        preds = np.squeeze(preds) if is_regression else np.argmax(preds, axis=1)
        if data_args.task_name is not None:
            result = metric.compute(predictions=preds, references=p.label_ids)
            if len(result) > 1:
                result["combined_score"] = np.mean(list(result.values())).item()
            return result
        elif is_regression:
            return {"mse": ((preds - p.label_ids) ** 2).mean().item()}
        else:
            micro_precision, micro_recall, micro_fb1, support = precision_recall_fscore_support(p.label_ids, preds, average='micro')
            macro_precision, macro_recall, macro_fb1, support = precision_recall_fscore_support(p.label_ids, preds, average='macro')
            weighted_precision, weighted_recall, weighted_fb1, support = precision_recall_fscore_support(p.label_ids, preds, average='weighted')
            sample_precision, sample_recall, sample_fb1, support = precision_recall_fscore_support(p.label_ids, preds, average=None)
            cfm = confusion_matrix(p.label_ids, preds) 

            return {"accuracy_direct": (preds == p.label_ids).astype(np.float32).mean().item(),
                'accuracy': accuracy_score(p.label_ids, preds),
                'micro-precision': micro_precision,
                'micro-recall': micro_recall,
                'micro-fb1': micro_fb1,
                'macro-precision': macro_precision,
                'macro-recall': macro_recall,
                'macro-fb1': macro_fb1,
                'support': support.tolist(),
                'sample-precision': sample_precision.tolist(),
                'sample-recall': sample_recall.tolist(), 
                'sample-fb1': sample_fb1.tolist(),
                'confusion_matrix': cfm.tolist()            
            }

    def compute_metrics_multiclass(p: EvalPrediction):
        preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
        preds_binary = (preds>0).astype(float)

        micro_precision, micro_recall, micro_fb1, support = precision_recall_fscore_support(p.label_ids, preds_binary, average='micro')
        macro_precision, macro_recall, macro_fb1, support = precision_recall_fscore_support(p.label_ids, preds_binary, average='macro')
        weighted_precision, weighted_recall, weighted_fb1, support = precision_recall_fscore_support(p.label_ids, preds_binary, average='weighted')
        sample_precision, sample_recall, sample_fb1, support = precision_recall_fscore_support(p.label_ids, preds_binary, average=None)
        confusion_matrix = multilabel_confusion_matrix(p.label_ids, preds_binary)

        return {'accuracy': accuracy_score(p.label_ids, preds_binary),
            'micro-precision': micro_precision,
            'micro-recall': micro_recall,
            'micro-fb1': micro_fb1,
            'macro-precision': macro_precision,
            'macro-recall': macro_recall,
            'macro-fb1': macro_fb1,
            'support': support.tolist(),
            'sample-precision': sample_precision.tolist(),
            'sample-recall': sample_recall.tolist(), 
            'sample-fb1': sample_fb1.tolist(),
            'confusion_matrix': confusion_matrix.tolist()
        }

    metric_computation_function = compute_metrics if not is_multilabel else compute_metrics_multiclass
    logging.info(f"Using metric_computation_function {metric_computation_function}")
    # Initialize our Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset if training_args.do_eval else None,
        compute_metrics=metric_computation_function,
        tokenizer=tokenizer,
        # Data collator will default to DataCollatorWithPadding, so we change it if we already did the padding.
        data_collator=default_data_collator if data_args.pad_to_max_length else None,
    )

    # Extra info - write labels to file
    output_label_file = os.path.join(training_args.output_dir, f"label_list.json")
    with open(output_label_file, "w") as writer:
        json.dump(label_to_id, writer, indent=4)    

    output_label_file = os.path.join(training_args.output_dir, f"training_args.json")
    with open(output_label_file, "w") as writer:
        writer.write(repr(training_args))

    output_label_file = os.path.join(training_args.output_dir, f"model_args.json")
    with open(output_label_file, "w") as writer:
        writer.write(repr(model_args))   

    output_label_file = os.path.join(training_args.output_dir, f"data_args.json")
    with open(output_label_file, "w") as writer:
        writer.write(repr(data_args))                          

    # Training
    if training_args.do_train:
        train_result = trainer.train(
            model_path=model_args.model_name_or_path if os.path.isdir(model_args.model_name_or_path) else None
        )
        metrics = train_result.metrics

        trainer.save_model()  # Saves the tokenizer too for easy upload

        # Save the training results to txt and json formats
        output_train_file = os.path.join(training_args.output_dir, "train_results.txt")
        output_train_file_json = os.path.join(training_args.output_dir, "train_results.json")
        if trainer.is_world_process_zero():
            with open(output_train_file, "w") as writer:
                logger.info("***** Train results *****")
                for key, value in sorted(metrics.items()):
                    logger.info(f"  {key} = {value}")
                    writer.write(f"{key} = {value}\n")
            with open(output_train_file_json, "w") as writer:
                json.dump(metrics, writer)

            # Need to save the state, since Trainer.save_model saves only the tokenizer with the model
            trainer.state.save_to_json(os.path.join(training_args.output_dir, "trainer_state.json"))

    # Evaluation
    eval_results = {}
    if training_args.do_eval:
        logger.info("*** Evaluate ***")

        # Loop to handle MNLI double evaluation (matched, mis-matched)
        tasks = [data_args.task_name]
        eval_datasets = [eval_dataset]
        if data_args.task_name == "mnli":
            tasks.append("mnli-mm")
            eval_datasets.append(datasets["validation_mismatched"])

        # Save the eval results to both txt and json formats
        for eval_dataset, task in zip(eval_datasets, tasks):
            eval_result = trainer.evaluate(eval_dataset=eval_dataset)

            output_eval_file = os.path.join(training_args.output_dir, f"eval_results_{task}.txt")
            output_eval_file_json = os.path.join(training_args.output_dir, f"eval_results_{task}.json")
            if trainer.is_world_process_zero():
                with open(output_eval_file, "w") as writer:
                    logger.info(f"***** Eval results {task} *****")
                    for key, value in sorted(eval_result.items()):
                        logger.info(f"  {key} = {value}")
                        writer.write(f"{key} = {value}\n")
                with open(output_eval_file_json, "w") as writer:
                    json.dump(eval_result, writer)
                
            eval_results.update(eval_result)

    if training_args.do_predict:
        logger.info("*** Test ***")

        # Loop to handle MNLI double evaluation (matched, mis-matched)
        tasks = [data_args.task_name]
        test_datasets = [test_dataset]
        if data_args.task_name == "mnli":
            tasks.append("mnli-mm")
            test_datasets.append(datasets["test_mismatched"])

        for test_dataset, task in zip(test_datasets, tasks):
            # Removing the `label` columns because it contains -1 and Trainer won't like that.
            test_dataset.remove_columns_(label_column_name)
            predictions = trainer.predict(test_dataset=test_dataset).predictions
            predictions = np.squeeze(predictions) if is_regression else np.argmax(predictions, axis=1)

            output_test_file = os.path.join(training_args.output_dir, f"test_results_{task}.txt")
            if trainer.is_world_process_zero():
                with open(output_test_file, "w") as writer:
                    logger.info(f"***** Test results {task} *****")
                    writer.write("index\tprediction\n")
                    for index, item in enumerate(predictions):
                        if is_regression:
                            writer.write(f"{index}\t{item:3.3f}\n")
                        else:
                            item = label_list[item]
                            writer.write(f"{index}\t{item}\n")

    return eval_results


def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()


if __name__ == "__main__":
    main()