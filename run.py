#!/usr/bin/env python

from json import load

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

dataset_arguments, model_arguments, training_arguments = HfArgumentParser(
    (DatasetArguments, ModelArguments, MyTrainingArguments)
).parse_args_into_dataclasses()
print(f"Device: {training_arguments.device}")
print(f"Training/evaluation parameters {training_arguments}")

set_verbosity_info()
enable_default_handler()
enable_explicit_format()

set_seed(training_arguments.seed)

with open("info.json") as fp:
    info = load(fp)[dataset_arguments.task_name]

datasets = load_dataset(info["path"], name=info["name"])

config = AutoConfig.from_pretrained(
    model_arguments.model_name_or_path,
    num_labels=info["num_labels"],
    finetuning_task=dataset_arguments.task_name,
)
tokenizer = AutoTokenizer.from_pretrained(model_arguments.model_name_or_path)
model_init = lambda: AutoModelForSequenceClassification.from_pretrained(
    model_arguments.model_name_or_path, config=config
)
model = model_init()

if dataset_arguments.task_name == "scitail":
    label = ClassLabel(names=("entails", "neutral"))

    def function(examples):
        examples["label"] = label.str2int(examples["label"])
        return examples

    datasets = datasets.map(function, batched=True)

if dataset_arguments.max_length > tokenizer.model_max_length:
    print(
        f"WARNING:{__name__}:The max_length passed ({dataset_arguments.max_length}) is larger than the maximum length for the model ({tokenizer.model_max_length}).",
        f"Using max_length={tokenizer.model_max_length}.",
    )
function = lambda examples: tokenizer(
    examples[info["text"]],
    text_pair=examples[info["text_pair"]],
    padding=dataset_arguments.padding,
    truncation=True,
    max_length=min(dataset_arguments.max_length, tokenizer.model_max_length),
)
datasets = datasets.map(function, batched=True)

datasets = datasets.filter(lambda example: example["label"] != -1)

metric = load_metric("accuracy")


def compute_metrics(eval_prediction):
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
    if dataset_arguments.padding == "max_length"
    else DataCollatorWithPadding(tokenizer, pad_to_multiple_of=8)
    if training_arguments.fp16
    else None
)

trainer = MyTrainer(
    model=None if training_arguments.do_hyperparameter_search else model,
    args=training_arguments,
    data_collator=data_collator,
    train_dataset=datasets[info["train_dataset"]],
    eval_dataset=datasets[info["eval_dataset"]] if training_arguments.do_eval else None,
    tokenizer=tokenizer,
    model_init=model_init if training_arguments.do_hyperparameter_search else None,
    compute_metrics=compute_metrics,
)

if training_arguments.do_hyperparameter_search:
    trainer.my_hyperparameter_search()

if training_arguments.do_train:
    trainer.my_train(model_arguments.model_name_or_path)

if training_arguments.do_eval:
    trainer.my_evaluate(datasets[info["eval_dataset"]])

if training_arguments.do_predict:
    if info["test_dataset"] is None:
        raise ValueError("Test dataset is empty.")
    trainer.my_predict(datasets[info["test_dataset"]])
