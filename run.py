#!/usr/bin/env python

from datasets import ClassLabel, load_dataset, load_metric

from args import DatasetArguments, ModelArguments, MyTrainingArguments
from trainer import MyTrainer
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
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
set_verbosity_info()
enable_default_handler()
enable_explicit_format()
print(f"Training/evaluation parameters {training_arguments}")

set_seed(training_arguments.seed)

datasets = (
    load_dataset("anli")
    if dataset_arguments.task_name.startswith("anli")
    else load_dataset(dataset_arguments.task_name, "tsv_format")
    if dataset_arguments.task_name == "scitail"
    else load_dataset(dataset_arguments.task_name)
    if dataset_arguments.task_name == "snli"
    else load_dataset("glue", dataset_arguments.task_name)
)

num_labels = (
    datasets["train_r1"].features["label"].num_classes
    if dataset_arguments.task_name.startswith("anli")
    else 2
    if dataset_arguments.task_name == "scitail"
    else datasets["train"].features["label"].num_classes
)

config = AutoConfig.from_pretrained(
    model_arguments.model_name_or_path,
    num_labels=num_labels,
    finetuning_task=dataset_arguments.task_name,
)
tokenizer = AutoTokenizer.from_pretrained(model_arguments.model_name_or_path)
if training_arguments.do_hyperparameter_search:
    model_init = lambda: AutoModelForSequenceClassification.from_pretrained(
        model_arguments.model_name_or_path,
        config=config,
        from_tf=bool(".ckpt" in model_arguments.model_name_or_path),
    )
else:
    model = AutoModelForSequenceClassification.from_pretrained(
        model_arguments.model_name_or_path,
        config=config,
        from_tf=bool(".ckpt" in model_arguments.model_name_or_path),
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

    def function(examples):
        examples["label"] = label.str2int(examples["label"])
        return examples

    datasets = datasets.map(function, batched=True)

datasets = datasets.map(
    lambda examples: tokenizer(
        examples[sentence1_key],
        text_pair=examples[sentence2_key],
        padding=padding,
        truncation=True,
        max_length=max_length,
    ),
    batched=True,
)

datasets = datasets.filter(lambda example: example["label"] != -1)

if dataset_arguments.task_name.startswith("anli"):
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


def compute_metrics(eval_prediction):
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
    trainer.my_predict(test_dataset)
