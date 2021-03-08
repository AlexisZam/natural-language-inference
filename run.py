#!/usr/bin/env python

import logging
import os
import random
import sys

import numpy as np
from datasets import ClassLabel, load_dataset, load_metric

import transformers
from args import DataTrainingArguments, ModelArguments
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    EvalPrediction,
    HfArgumentParser,
    PretrainedConfig,
    Trainer,
    TrainingArguments,
    default_data_collator,
    set_seed,
)
from transformers.trainer_utils import get_last_checkpoint

logger = logging.getLogger(__name__)

# See all possible arguments in src/transformers/training_args.py
# or by passing the --help flag to this script.

parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
model_args, data_args, training_args = parser.parse_args_into_dataclasses()

# Detecting last checkpoint.
last_checkpoint = None
if (
    os.path.isdir(training_args.output_dir)
    and training_args.do_train
    and not training_args.overwrite_output_dir
):
    last_checkpoint = get_last_checkpoint(training_args.output_dir)
    if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
        raise ValueError(
            f"Output directory ({training_args.output_dir}) already exists and is not empty. Use --overwrite_output_dir to overcome."
        )
    elif last_checkpoint is not None:
        logger.info(
            f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
        )

# Setup logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger.setLevel(logging.INFO)

# Log on each process the small summary:
logger.warning(f"Device: {training_args.device}16-bits training: {training_args.fp16}")
# Set the verbosity to info of the Transformers logger:
transformers.utils.logging.set_verbosity_info()
transformers.utils.logging.enable_default_handler()
transformers.utils.logging.enable_explicit_format()
logger.info(f"Training/evaluation parameters {training_args}")

# Set seed before initializing model.
set_seed(training_args.seed)

# Get the datasets.
# Downloading and loading a dataset from the hub.
datasets = (
    load_dataset("anli")
    if data_args.task_name.startswith("anli")
    else load_dataset(data_args.task_name, "tsv_format")
    if data_args.task_name == "scitail"
    else load_dataset(data_args.task_name)
    if data_args.task_name == "snli"
    else load_dataset("glue", data_args.task_name)
)

# Labels
label_list = (
    datasets["train_r1"].features["label"].names
    if data_args.task_name.startswith("anli")
    else ("entails", "neutral")
    if data_args.task_name == "scitail"
    else datasets["train"].features["label"].names
)
num_labels = len(label_list)

# Load pretrained model and tokenizer
config = AutoConfig.from_pretrained(
    model_args.config_name if model_args.config_name else model_args.model_name_or_path,
    num_labels=num_labels,
    finetuning_task=data_args.task_name,
    cache_dir=model_args.cache_dir,
    revision=model_args.model_revision,
    use_auth_token=True if model_args.use_auth_token else None,
)
tokenizer = AutoTokenizer.from_pretrained(
    model_args.tokenizer_name
    if model_args.tokenizer_name
    else model_args.model_name_or_path,
    cache_dir=model_args.cache_dir,
    use_fast=model_args.use_fast_tokenizer,
    revision=model_args.model_revision,
    use_auth_token=True if model_args.use_auth_token else None,
)
model = AutoModelForSequenceClassification.from_pretrained(
    model_args.model_name_or_path,
    from_tf=bool(".ckpt" in model_args.model_name_or_path),
    config=config,
    cache_dir=model_args.cache_dir,
    revision=model_args.model_revision,
    use_auth_token=True if model_args.use_auth_token else None,
)

# Preprocessing the datasets
sentence1_key, sentence2_key = (
    ("question", "sentence")
    if data_args.task_name == "qnli"
    else ("sentence1", "sentence2")
    if data_args.task_name in ("rte", "wnli")
    else ("premise", "hypothesis")
)

# Padding strategy
padding = "max_length" if data_args.pad_to_max_length else False

if data_args.max_seq_length > tokenizer.model_max_length:
    logger.warn(
        f"The max_seq_length passed ({data_args.max_seq_length}) is larger than the maximum length for the model ({tokenizer.model_max_length}). Using max_seq_length={tokenizer.model_max_length}."
    )
max_seq_length = min(data_args.max_seq_length, tokenizer.model_max_length)

if data_args.task_name == "scitail":
    label = ClassLabel(names=label_list)


def preprocess_function(examples):
    # Tokenize the texts
    result = tokenizer(
        examples[sentence1_key],
        examples[sentence2_key],
        padding=padding,
        max_length=max_seq_length,
        truncation=True,
    )

    if data_args.task_name == "scitail":
        result["label"] = label.str2int(examples["label"])

    # Map labels to IDs (not necessary for GLUE tasks)
    if label_to_id is not None and "label" in examples:
        result["label"] = [
            (label_to_id[l] if l != -1 else -1) for l in examples["label"]
        ]
    return result


datasets = datasets.map(
    preprocess_function,
    batched=True,
    load_from_cache_file=not data_args.overwrite_cache,
)

if data_args.task_name == "snli":
    datasets = datasets.filter(
        lambda example: example["label"] != -1,
        load_from_cache_file=not data_args.overwrite_cache,
    )

if data_args.task_name.startswith("anli"):
    round = data_args.task_name.split("_")[1]
    train_dataset = datasets[f"train_{round}"]
    eval_dataset = datasets[f"dev_{round}"]
    test_dataset = datasets[f"test_{round}"]
else:
    train_dataset = datasets["train"]
    eval_dataset = datasets[
        "validation_matched" if data_args.task_name == "mnli" else "validation"
    ]
    test_dataset = datasets["test_matched" if data_args.task_name == "mnli" else "test"]

# Log a few random samples from the training set:
for index in random.sample(range(len(train_dataset)), 3):
    logger.info(f"Sample {index} of the training set: {train_dataset[index]}.")

# Get the metric function
metric = load_metric("accuracy")


def compute_metrics(p: EvalPrediction):
    preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
    preds = np.argmax(preds, axis=1)
    result = metric.compute(predictions=preds, references=p.label_ids)
    if len(result) > 1:
        result["combined_score"] = np.mean(list(result.values())).item()
    return result


# Data collator will default to DataCollatorWithPadding, so we change it if we already did the padding.
data_collator = (
    default_data_collator
    if data_args.pad_to_max_length
    else DataCollatorWithPadding(tokenizer, pad_to_multiple_of=8)
    if training_args.fp16
    else None
)

# Initialize our Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset if training_args.do_eval else None,
    compute_metrics=compute_metrics,
    tokenizer=tokenizer,
    data_collator=data_collator,
)

# Training
if training_args.do_train:
    checkpoint = (
        last_checkpoint
        if last_checkpoint is not None
        else model_args.model_name_or_path
        if os.path.isdir(model_args.model_name_or_path)
        else None
    )
    train_result = trainer.train(resume_from_checkpoint=checkpoint)
    metrics = train_result.metrics

    trainer.save_model()  # Saves the tokenizer too for easy upload

    output_train_file = os.path.join(training_args.output_dir, "train_results.txt")
    with open(output_train_file, "w") as writer:
        logger.info("***** Train results *****")
        for key, value in sorted(metrics.items()):
            logger.info(f"  {key} = {value}")
            writer.write(f"{key} = {value}\n")

    # Need to save the state, since Trainer.save_model saves only the tokenizer with the model
    trainer.state.save_to_json(
        os.path.join(training_args.output_dir, "trainer_state.json")
    )

# Evaluation
if training_args.do_eval:
    logger.info("*** Evaluate ***")

    # Loop to handle MNLI double evaluation (matched, mis-matched)
    tasks = [data_args.task_name]
    eval_datasets = [eval_dataset]
    if data_args.task_name == "mnli":
        tasks.append("mnli-mm")
        eval_datasets.append(datasets["validation_mismatched"])

    for eval_dataset, task in zip(eval_datasets, tasks):
        eval_result = trainer.evaluate(eval_dataset=eval_dataset)

        output_eval_file = os.path.join(
            training_args.output_dir, f"eval_results_{task}.txt"
        )
        with open(output_eval_file, "w") as writer:
            logger.info(f"***** Eval results {task} *****")
            for key, value in sorted(eval_result.items()):
                logger.info(f"  {key} = {value}")
                writer.write(f"{key} = {value}\n")

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
        test_dataset.remove_columns_("label")
        predictions = trainer.predict(test_dataset=test_dataset).predictions
        predictions = np.argmax(predictions, axis=1)

        output_test_file = os.path.join(
            training_args.output_dir, f"test_results_{task}.txt"
        )
        with open(output_test_file, "w") as writer:
            logger.info(f"***** Test results {task} *****")
            writer.write("index\tprediction\n")
            for index, item in enumerate(predictions):
                item = label_list[item]
                writer.write(f"{index}\t{item}\n")

if False:
    # Hyperparameter search
    def model_init():
        return AutoModelForSequenceClassification.from_pretrained(
            model_args.model_name_or_path, num_labels=num_labels
        )

    trainer = Trainer(
        model_init=model_init,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset if training_args.do_eval else None,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )

    best_run = trainer.hyperparameter_search(n_trials=10, direction="maximize")

    for n, v in best_run.hyperparameters.items():
        setattr(trainer.args, n, v)

    trainer.train()
