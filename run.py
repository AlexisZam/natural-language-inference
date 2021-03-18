#!/usr/bin/env python

from datasets import ClassLabel, load_dataset, load_metric

from args import DatasetArguments, ModelArguments, MyTrainingArguments
from dataset_info_dict import dataset_info_dict
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
# print(f"Device: {training_arguments.device}")
# print(f"Training/evaluation parameters {training_arguments}")

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

# Data collator will default to DataCollatorWithPadding, so we change it if we already did the padding.
data_collator = (
    default_data_collator
    if dataset_arguments.padding == "max_length"
    else DataCollatorWithPadding(tokenizer, pad_to_multiple_of=8)
    if training_arguments.fp16
    else None
)  # FIXME

dataset_info = dataset_info_dict[dataset_arguments.dataset_name]
# if dataset_arguments.max_length > tokenizer.model_max_length:
#     print(
#         f"WARNING:{__name__}:The max_length passed ({dataset_arguments.max_length}) is larger than the maximum length for the model ({tokenizer.model_max_length}).",
#         f"Using max_length={tokenizer.model_max_length}.",
#     )
function = lambda batch: tokenizer(
    batch[dataset_info["text"]],
    text_pair=batch[dataset_info["text_pair"]],
    padding=dataset_arguments.padding,
    truncation=True,
    max_length=min(dataset_arguments.max_length, tokenizer.model_max_length),
)  # FIXME
dataset_dict = (
    load_dataset(dataset_info["path"], name=dataset_info["name"])
    .filter(lambda example: example["label"] != -1)
    .map(function, batched=True)
)
# if dataset_arguments.dataset_name == "scitail":
#     label = ClassLabel(names=("entails", "neutral"))

#     def function(examples):
#         examples["label"] = label.str2int(examples["label"])
#         return examples

#     dataset_dict = dataset_dict.map(function, batched=True)

metric = load_metric("accuracy")
compute_metrics = lambda eval_prediction: metric.compute(
    predictions=(
        eval_prediction.predictions[0]
        if isinstance(eval_prediction.predictions, tuple)
        else eval_prediction.predictions
    ).argmax(axis=1),
    references=eval_prediction.label_ids,
)  # FIXME

trainer = MyTrainer(
    model=None if training_arguments.do_hyperparameter_search else model_init(),
    args=training_arguments,
    data_collator=data_collator,
    train_dataset=dataset_dict[dataset_info["train_dataset"]],
    eval_dataset=dataset_dict[dataset_info["eval_dataset"]]
    if training_arguments.do_eval
    else None,  # FIXME
    tokenizer=tokenizer,
    model_init=model_init if training_arguments.do_hyperparameter_search else None,
    compute_metrics=compute_metrics,
)

if training_arguments.do_hyperparameter_search:
    trainer.my_hyperparameter_search()

if training_arguments.do_train:
    trainer.my_train(model_arguments.pretrained_model_name_or_path)

if training_arguments.do_eval:
    trainer.my_evaluate()

if training_arguments.do_predict:
    # if dataset_info["test_dataset"] is None:
    #     raise ValueError("Test dataset is empty.")
    trainer.my_predict(dataset_dict[dataset_info["test_dataset"]])  # FIXME
