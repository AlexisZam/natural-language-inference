#!/usr/bin/env python

from datasets import load_metric
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    HfArgumentParser,
    default_data_collator,
    set_seed,
)
from transformers.utils.logging import (
    enable_default_handler,
    enable_explicit_format,
    set_verbosity_info,
)

from args import DatasetArguments, ModelArguments, MyTrainingArguments
from load_dataset import my_load_dataset
from trainer import MyTrainer

enable_default_handler()
enable_explicit_format()
set_verbosity_info()

dataset_arguments, model_arguments, training_arguments = HfArgumentParser(
    (DatasetArguments, ModelArguments, MyTrainingArguments)
).parse_args_into_dataclasses()

set_seed(training_arguments.seed)

tokenizer = AutoTokenizer.from_pretrained(model_arguments.pretrained_model_name)

dataset_dict = my_load_dataset(dataset_arguments, tokenizer)

num_labels = (
    2
    if dataset_arguments.dataset_name == "scitail"
    else dataset_dict["train"].features["label"].num_classes
)
config = AutoConfig.from_pretrained(
    model_arguments.pretrained_model_name, num_labels=num_labels
)
model_init = lambda: AutoModelForSequenceClassification.from_pretrained(
    model_arguments.pretrained_model_name, config=config
)

metric = load_metric("accuracy")
compute_metrics = lambda eval_prediction: metric.compute(
    predictions=(eval_prediction.predictions).argmax(axis=1),
    references=eval_prediction.label_ids,
)

trainer = MyTrainer(
    model=None if training_arguments.do_hyperparameter_search else model_init(),
    args=training_arguments,
    data_collator=default_data_collator,
    train_dataset=dataset_dict["train"],
    eval_dataset=dataset_dict["eval"] if training_arguments.do_eval else None,
    tokenizer=tokenizer,
    model_init=model_init if training_arguments.do_hyperparameter_search else None,
    compute_metrics=compute_metrics,
)

if training_arguments.do_hyperparameter_search:
    trainer.my_hyperparameter_search()

if training_arguments.do_train:
    trainer.my_train()

if training_arguments.do_eval:
    trainer.my_evaluate()

if training_arguments.do_predict:
    trainer.my_predict(dataset_dict["test"])
