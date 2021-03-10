#!/usr/bin/env python

import logging
import random
import sys
from pathlib import Path

import numpy as np
from datasets import ClassLabel, load_dataset, load_metric

import transformers
from args import DataTrainingArguments, ModelArguments
from trainer import MyTrainer
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    EvalPrediction,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    default_data_collator,
    set_seed,
)
from transformers.trainer_utils import get_last_checkpoint


def main():
    logger = logging.getLogger(__name__)

    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.

    parser = HfArgumentParser(
        (ModelArguments, DataTrainingArguments, TrainingArguments)
    )
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # Detecting last checkpoint.
    last_checkpoint = None
    if (
        Path(training_args.output_dir).is_dir()
        and training_args.do_train
        and not training_args.overwrite_output_dir
    ):
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and any(Path(training_args.output_dir).iterdir()):
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None:
            print(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    # Log on each process the small summary:
    logger.warning(
        f"Device: {training_args.device}16-bits training: {training_args.fp16}"
    )
    # Set the verbosity to info of the Transformers logger:
    transformers.utils.logging.set_verbosity_info()
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()
    print(f"Training/evaluation parameters {training_args}")

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
        model_args.config_name
        if model_args.config_name
        else model_args.model_name_or_path,
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
        test_dataset = datasets[
            "test_matched" if data_args.task_name == "mnli" else "test"
        ]

    # Log a few random samples from the training set:
    for index in random.sample(range(len(train_dataset)), 3):
        print(f"Sample {index} of the training set: {train_dataset[index]}.")

    # Get the metric function
    metric = load_metric("accuracy")

    def compute_metrics(p: EvalPrediction):
        preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
        preds = preds.argmax(axis=1)
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
    trainer = MyTrainer(
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
        trainer.my_train(
            last_checkpoint, model_args.model_name_or_path, training_args.output_dir
        )

    # Evaluation
    if training_args.do_eval:
        trainer.my_evaluate(
            data_args.task_name, eval_dataset, datasets, training_args.output_dir
        )

    if training_args.do_predict:
        trainer.my_predict(
            data_args.task_name,
            test_dataset,
            datasets,
            training_args.output_dir,
            label_list,
        )

    # Hyperparameter search
    if False:
        model_init = lambda: AutoModelForSequenceClassification.from_pretrained(
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


main()
