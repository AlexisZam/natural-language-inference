#!/usr/bin/env python

from datasets import load_metric

from args import DatasetArguments, ModelArguments, MyTrainingArguments
from load_dataset import my_load_dataset
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

enable_default_handler()
enable_explicit_format()
set_verbosity_info()

dataset_arguments, model_arguments, training_arguments = HfArgumentParser(
    (DatasetArguments, ModelArguments, MyTrainingArguments)
).parse_args_into_dataclasses()

set_seed(training_arguments.seed)

# config = AutoConfig.from_pretrained(
#     model_arguments.pretrained_model_name_or_path,
#     num_labels=dataset_info["num_labels"],
#     finetuning_task=dataset_arguments.dataset_name,
# )
model_init = lambda: AutoModelForSequenceClassification.from_pretrained(
    model_arguments.pretrained_model_name_or_path  # , config=config
)

tokenizer = AutoTokenizer.from_pretrained(model_arguments.pretrained_model_name_or_path)

# FIXME
# Data collator will default to DataCollatorWithPadding, so we change it if we already did the padding.
data_collator = (
    default_data_collator
    if dataset_arguments.padding == "max_length"
    else DataCollatorWithPadding(tokenizer, pad_to_multiple_of=8)
    if training_arguments.fp16
    else None
)

dataset_dict = my_load_dataset(dataset_arguments, tokenizer)  # FIXME

metric = load_metric("accuracy")
compute_metrics = lambda eval_prediction: metric.compute(
    predictions=(eval_prediction.predictions).argmax(axis=1),
    references=eval_prediction.label_ids,
)

trainer = MyTrainer(
    model=None if training_arguments.do_hyperparameter_search else model_init(),
    args=training_arguments,
    data_collator=data_collator,
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
