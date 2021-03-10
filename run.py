#!/usr/bin/env python

from re import compile

import numpy as np
from datasets import ClassLabel, load_dataset, load_metric

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
from transformers.utils.logging import (
    enable_default_handler,
    enable_explicit_format,
    set_verbosity_info,
)


def main():
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.

    parser = HfArgumentParser(
        (ModelArguments, DataTrainingArguments, TrainingArguments)
    )
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # Log on each process the small summary:
    print(f"Device: {training_args.device}")
    # Set the verbosity to info of the Transformers logger:
    set_verbosity_info()
    enable_default_handler()
    enable_explicit_format()
    print(f"Training/evaluation parameters {training_args}")

    # Set seed before initializing model.
    set_seed(training_args.seed)

    anli_pattern = compile("^anli_r[1-3]$")

    # Get the datasets.
    datasets = (
        load_dataset("anli")
        if anli_pattern.search(data_args.task_name)
        else load_dataset(data_args.task_name, "tsv_format")
        if data_args.task_name == "scitail"
        else load_dataset(data_args.task_name)
        if data_args.task_name == "snli"
        else load_dataset("glue", data_args.task_name)
    )

    # Labels
    label_list = (
        datasets["train_r1"].features["label"].names
        if anli_pattern.search(data_args.task_name)
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
        print(
            f"WARNING:{__name__}:The max_seq_length passed ({data_args.max_seq_length}) is larger than the maximum length for the model ({tokenizer.model_max_length}). Using max_seq_length={tokenizer.model_max_length}."
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

    datasets = datasets.remove_columns(
        "idx" if data_args.task_name == "qnli" else ("idx", "sentence1", "sentence2")
    )

    if data_args.task_name == "snli":
        datasets = datasets.filter(
            lambda example: example["label"] != -1,
            load_from_cache_file=not data_args.overwrite_cache,
        )

    if anli_pattern.search(data_args.task_name):
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
        trainer.my_train(model_args.model_name_or_path)

    # Evaluation
    if training_args.do_eval:
        trainer.my_evaluate(data_args.task_name, eval_dataset, datasets)

    if training_args.do_predict:
        trainer.my_predict(
            data_args.task_name,
            test_dataset,
            datasets,
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
