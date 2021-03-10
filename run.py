#!/usr/bin/env python

from re import compile

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

parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
(
    model_arguments,
    data_training_arguments,
    training_arguments,
) = parser.parse_args_into_dataclasses()

# Log on each process the small summary:
print(f"Device: {training_arguments.device}")
# Set the verbosity to info of the Transformers logger:
set_verbosity_info()
enable_default_handler()
enable_explicit_format()
print(f"Training/evaluation parameters {training_arguments}")

# Set seed before initializing model.
set_seed(training_arguments.seed)

anli_pattern = compile("^anli_r[1-3]$")

datasets = (
    load_dataset("anli")
    if anli_pattern.search(data_training_arguments.task_name) is not None
    else load_dataset(data_training_arguments.task_name, "tsv_format")
    if data_training_arguments.task_name == "scitail"
    else load_dataset(data_training_arguments.task_name)
    if data_training_arguments.task_name == "snli"
    else load_dataset("glue", data_training_arguments.task_name)
)

label_list = (
    datasets["train_r1"].features["label"].names
    if anli_pattern.search(data_training_arguments.task_name) is not None
    else ("entails", "neutral")
    if data_training_arguments.task_name == "scitail"
    else datasets["train"].features["label"].names
)
num_labels = len(label_list)

config = AutoConfig.from_pretrained(
    model_arguments.config_name
    if model_arguments.config_name
    else model_arguments.model_name_or_path,
    num_labels=num_labels,
    finetuning_task=data_training_arguments.task_name,
    cache_dir=model_arguments.cache_dir,
    revision=model_arguments.model_revision,
    use_auth_token=True if model_arguments.use_auth_token else None,
)
tokenizer = AutoTokenizer.from_pretrained(
    model_arguments.tokenizer_name
    if model_arguments.tokenizer_name
    else model_arguments.model_name_or_path,
    cache_dir=model_arguments.cache_dir,
    use_fast=model_arguments.use_fast_tokenizer,
    revision=model_arguments.model_revision,
    use_auth_token=True if model_arguments.use_auth_token else None,
)
model = AutoModelForSequenceClassification.from_pretrained(
    model_arguments.model_name_or_path,
    from_tf=bool(".ckpt" in model_arguments.model_name_or_path),
    config=config,
    cache_dir=model_arguments.cache_dir,
    revision=model_arguments.model_revision,
    use_auth_token=True if model_arguments.use_auth_token else None,
)

sentence1_key, sentence2_key = (
    ("question", "sentence")
    if data_training_arguments.task_name == "qnli"
    else ("sentence1", "sentence2")
    if data_training_arguments.task_name in ("rte", "wnli")
    else ("premise", "hypothesis")
)

padding = "max_length" if data_training_arguments.pad_to_max_length else False

if data_training_arguments.max_seq_length > tokenizer.model_max_length:
    print(
        f"WARNING:{__name__}:The max_seq_length passed ({data_training_arguments.max_seq_length}) is larger than the maximum length for the model ({tokenizer.model_max_length}).",
        f"Using max_seq_length={tokenizer.model_max_length}.",
    )
max_seq_length = min(data_training_arguments.max_seq_length, tokenizer.model_max_length)

if data_training_arguments.task_name == "scitail":
    label = ClassLabel(names=label_list)


def preprocess_function(examples):
    result = tokenizer(
        examples[sentence1_key],
        examples[sentence2_key],
        padding=padding,
        max_length=max_seq_length,
        truncation=True,
    )

    if data_training_arguments.task_name == "scitail":
        result["label"] = label.str2int(examples["label"])

    return result


datasets = datasets.map(
    preprocess_function,
    batched=True,
    load_from_cache_file=data_training_arguments.load_from_cache_file,
)

datasets = datasets.remove_columns(
    "idx"
    if data_training_arguments.task_name == "qnli"
    else ("idx", "sentence1", "sentence2")
)

if data_training_arguments.task_name == "snli":
    datasets = datasets.filter(
        lambda example: example["label"] != -1,
        load_from_cache_file=data_training_arguments.load_from_cache_file,
    )

if anli_pattern.search(data_training_arguments.task_name) is not None:
    round = data_training_arguments.task_name.split("_")[1]
    train_dataset = datasets[f"train_{round}"]
    eval_dataset = datasets[f"dev_{round}"]
    test_dataset = datasets[f"test_{round}"]
else:
    train_dataset = datasets["train"]
    eval_dataset = datasets[
        "validation_matched"
        if data_training_arguments.task_name == "mnli"
        else "validation"
    ]
    test_dataset = datasets[
        "test_matched" if data_training_arguments.task_name == "mnli" else "test"
    ]

metric = load_metric("accuracy")


def compute_metrics(eval_prediction: EvalPrediction):
    predictions = (
        eval_prediction.predictions[0]
        if isinstance(eval_prediction.predictions, tuple)
        else eval_prediction.predictions
    )
    predictions = predictions.argmax(axis=1)
    return metric.compute(predictions=predictions, references=eval_prediction.label_ids)


# Data collator will default to DataCollatorWithPadding, so we change it if we already did the padding.
data_collator = (
    default_data_collator
    if data_training_arguments.pad_to_max_length
    else DataCollatorWithPadding(tokenizer, pad_to_multiple_of=8)
    if training_arguments.fp16
    else None
)

trainer = MyTrainer(
    model=model,
    args=training_arguments,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset if training_arguments.do_eval else None,
    compute_metrics=compute_metrics,
    tokenizer=tokenizer,
    data_collator=data_collator,
)

if False:
    model_init = lambda: AutoModelForSequenceClassification.from_pretrained(
        model_arguments.model_name_or_path, num_labels=num_labels
    )

    trainer = MyTrainer(
        model_init=model_init,
        args=training_arguments,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset if training_arguments.do_eval else None,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )

    trainer.my_hyperparameter_search()

if training_arguments.do_train:
    trainer.my_train(model_arguments.model_name_or_path)

if training_arguments.do_eval:
    trainer.my_evaluate(data_training_arguments.task_name, eval_dataset, datasets)

if training_arguments.do_predict:
    trainer.my_predict(
        data_training_arguments.task_name, test_dataset, datasets, label_list
    )
