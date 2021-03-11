#!/usr/bin/env python

from re import compile

from datasets import ClassLabel, load_dataset, load_metric

from args import DatasetArguments, ModelArguments, MyTrainingArguments
from trainer import MyTrainer
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    EvalPrediction,
    HfArgumentParser,
    default_data_collator,
    set_seed,
)
from transformers.utils.logging import (
    enable_default_handler,
    enable_explicit_format,
    set_verbosity_info,
)

parser = HfArgumentParser((DatasetArguments, ModelArguments, MyTrainingArguments))
(
    dataset_arguments,
    model_arguments,
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
    if anli_pattern.search(dataset_arguments.task_name) is not None
    else load_dataset(dataset_arguments.task_name, "tsv_format")
    if dataset_arguments.task_name == "scitail"
    else load_dataset(dataset_arguments.task_name)
    if dataset_arguments.task_name == "snli"
    else load_dataset("glue", dataset_arguments.task_name)
)

num_labels = (
    datasets["train_r1"].features["label"].num_classes
    if anli_pattern.search(dataset_arguments.task_name) is not None
    else 2
    if dataset_arguments.task_name == "scitail"
    else datasets["train"].features["label"].num_classes
)

config = AutoConfig.from_pretrained(
    model_arguments.config_name
    if model_arguments.config_name
    else model_arguments.model_name_or_path,
    revision=model_arguments.revision,
    num_labels=num_labels,
    finetuning_task=dataset_arguments.task_name,
    use_auth_token=True if model_arguments.use_auth_token else None,
)
tokenizer = AutoTokenizer.from_pretrained(
    model_arguments.tokenizer_name
    if model_arguments.tokenizer_name
    else model_arguments.model_name_or_path,
    revision=model_arguments.revision,
    use_fast=model_arguments.use_fast,
    use_auth_token=True if model_arguments.use_auth_token else None,
)
if training_arguments.do_hyperparameter_search:
    model_init = lambda: AutoModelForSequenceClassification.from_pretrained(
        model_arguments.model_name_or_path,
        config=config,
        from_tf=bool(".ckpt" in model_arguments.model_name_or_path),
        revision=model_arguments.revision,
        use_auth_token=True if model_arguments.use_auth_token else None,
    )
else:
    model = AutoModelForSequenceClassification.from_pretrained(
        model_arguments.model_name_or_path,
        config=config,
        from_tf=bool(".ckpt" in model_arguments.model_name_or_path),
        revision=model_arguments.revision,
        use_auth_token=True if model_arguments.use_auth_token else None,
    )

sentence1_key, sentence2_key = (
    ("question", "sentence")
    if dataset_arguments.task_name == "qnli"
    else ("sentence1", "sentence2")
    if dataset_arguments.task_name in ("rte", "wnli")
    else ("premise", "hypothesis")
)

padding = "max_length" if dataset_arguments.pad_to_max_length else False

if dataset_arguments.max_length > tokenizer.model_max_length:
    print(
        f"WARNING:{__name__}:The max_length passed ({dataset_arguments.max_length}) is larger than the maximum length for the model ({tokenizer.model_max_length}).",
        f"Using max_length={tokenizer.model_max_length}.",
    )
max_length = min(dataset_arguments.max_length, tokenizer.model_max_length)

if dataset_arguments.task_name == "scitail":
    label = ClassLabel(names=("entails", "neutral"))


def preprocess_function(examples):
    result = tokenizer(
        examples[sentence1_key],
        text_pair=examples[sentence2_key],
        padding=padding,
        truncation=True,
        max_length=max_length,
    )

    if dataset_arguments.task_name == "scitail":
        result["label"] = label.str2int(examples["label"])

    return result


datasets = datasets.map(
    preprocess_function,
    batched=True,
    load_from_cache_file=dataset_arguments.load_from_cache_file,
)

datasets = datasets.filter(
    lambda example: example["label"] != -1,
    load_from_cache_file=dataset_arguments.load_from_cache_file,
)

if anli_pattern.search(dataset_arguments.task_name) is not None:
    round = dataset_arguments.task_name.split("_")[1]
    train_dataset = datasets[f"train_{round}"]
    eval_dataset = datasets[f"dev_{round}"]
    test_dataset = datasets[f"test_{round}"]
else:
    train_dataset = datasets["train"]
    eval_dataset = datasets[
        "validation_matched" if dataset_arguments.task_name == "mnli" else "validation"
    ]
    test_dataset = datasets[
        "test_matched" if dataset_arguments.task_name == "mnli" else "test"
    ]

metric = load_metric("accuracy")


def compute_metrics(eval_prediction: EvalPrediction):
    predictions = (
        eval_prediction.predictions[0]
        if isinstance(eval_prediction.predictions, tuple)
        else eval_prediction.predictions
    ).argmax(axis=1)
    return metric.compute(predictions=predictions, references=eval_prediction.label_ids)


# Data collator will default to DataCollatorWithPadding, so we change it if we already did the padding.
data_collator = (
    default_data_collator
    if dataset_arguments.pad_to_max_length
    else DataCollatorWithPadding(tokenizer, pad_to_multiple_of=8)
    if training_arguments.fp16
    else None
)

trainer = MyTrainer(
    model=None if training_arguments.do_hyperparameter_search else model,
    args=training_arguments,
    data_collator=data_collator,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset if training_arguments.do_eval else None,
    tokenizer=tokenizer,
    model_init=model_init if training_arguments.do_hyperparameter_search else None,
    compute_metrics=compute_metrics,
)

if training_arguments.do_hyperparameter_search:
    trainer.my_hyperparameter_search()

if training_arguments.do_train:
    trainer.my_train(model_arguments.model_name_or_path)

if training_arguments.do_eval:
    trainer.my_evaluate(dataset_arguments.task_name, eval_dataset, datasets)

if training_arguments.do_predict:
    trainer.my_predict(dataset_arguments.task_name, test_dataset, datasets)
