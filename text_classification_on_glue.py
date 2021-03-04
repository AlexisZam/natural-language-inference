#!/usr/bin/env python

import numpy as np
from datasets import load_dataset, load_metric

from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)

"""- [MNLI](https://arxiv.org/abs/1704.05426) (Multi-Genre Natural Language
  Inference) Determine if a sentence entails, contradicts or is unrelated to a
  given hypothesis. (This dataset has two versions, one with the validation and
  test set coming from the same distribution, another called mismatched where
  the validation and test use out-of-domain data.)
- [QNLI](https://rajpurkar.github.io/SQuAD-explorer/) (Question-answering
  Natural Language Inference) Determine if the answer to a question is in the
  second sentence or not. (This dataset is built from the SQuAD dataset.)
- [RTE](https://aclweb.org/aclwiki/Recognizing_Textual_Entailment) (Recognizing
  Textual Entailment) Determine if a sentence entails a given hypothesis or not.
- [WNLI](https://cs.nyu.edu/faculty/davise/papers/WinogradSchemas/WS.html)
  (Winograd Natural Language Inference) Determine if a sentence with an
  anonymous pronoun and a sentence with this pronoun replaced are entailed or
  not. (This dataset is built from the Winograd Schema Challenge dataset.)

This notebook is built to run on any of the tasks in the list above, with any
model checkpoint from the [Model Hub](https://huggingface.co/models) as long as
that model has a version with a classification head. Depending on you model and
the GPU you are using, you might need to adjust the batch size to avoid
out-of-memory errors. Set those three parameters, then the rest of the notebook
should run smoothly."""

task = "rte"
model_checkpoint = "distilbert-base-uncased"
batch_size = 16

# Loading the dataset

actual_task = "mnli" if task == "mnli-mm" else task
dataset = load_dataset("glue", actual_task)
metric = load_metric("accuracy")

# Preprocessing the data

"""A 🤗 Transformers `Tokenizer` will tokenize the inputs (including converting
the tokens to their corresponding IDs in the pretrained vocabulary) and put it
in a format the model expects, as well as generate the other inputs that model
requires.

To do all of this, we instantiate our tokenizer with the
`AutoTokenizer.from_pretrained` method, which will ensure:

- we get a tokenizer that corresponds to the model architecture we want to use,
- we download the vocabulary used when pretraining this specific checkpoint.
"""

tokenizer = AutoTokenizer.from_pretrained(model_checkpoint, use_fast=True)

task_to_keys = {
    "mnli": ("premise", "hypothesis"),
    "mnli-mm": ("premise", "hypothesis"),
    "qnli": ("question", "sentence"),
    "rte": ("sentence1", "sentence2"),
    "wnli": ("sentence1", "sentence2"),
}

sentence1_key, sentence2_key = task_to_keys[task]

"""We just feed our samples to the `tokenizer` with the argument
`truncation=True`. This will ensure that an input longer that what the model
selected can handle will be truncated to the maximum length accepted by the
model."""


def preprocess_function(examples):
    return tokenizer(examples[sentence1_key], examples[sentence2_key], truncation=True)


encoded_dataset = dataset.map(preprocess_function, batched=True)

# Fine-tuning the model

num_labels = 3 if task.startswith("mnli") else 2
model = AutoModelForSequenceClassification.from_pretrained(
    model_checkpoint, num_labels=num_labels
)

"""The warning is telling us we are throwing away some weights (the
`vocab_transform` and `vocab_layer_norm` layers) and randomly initializing some
other (the `pre_classifier` and `classifier` layers). This is absolutely normal
in this case, because we are removing the head used to pretrain the model on a
masked language modeling objective and replacing it with a new head for which we
don't have pretrained weights, so the library warns us we should fine-tune this
model before using it for inference, which is exactly what we are going to do.
"""

metric_name = "accuracy"

args = TrainingArguments(
    "test-glue",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    num_train_epochs=5,
    weight_decay=0.01,
    load_best_model_at_end=True,
    metric_for_best_model=metric_name,
)

"""The only preprocessing we have to do is to take the argmax of our predicted
logits.
"""


def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return metric.compute(predictions=predictions, references=labels)


validation_key = (
    "validation_mismatched"
    if task == "mnli-mm"
    else "validation_matched"
    if task == "mnli"
    else "validation"
)
trainer = Trainer(
    model,
    args,
    train_dataset=encoded_dataset["train"],
    eval_dataset=encoded_dataset[validation_key],
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)

"""You might wonder why we pass along the `tokenizer` when we already
preprocessed our data. This is because we will use it once last time to make all
the samples we gather the same length by applying padding, which requires
knowing the model's preferences regarding padding (to the left or right? with
which token?). The `tokenizer` has a pad method that will do all of this right
for us, and the `Trainer` will use it. You can customize this part by defining
and passing your own `data_collator` which will receive the samples like the
dictionaries seen above and will need to return a dictionary of tensors.
"""

trainer.train()

"""We can check with the `evaluate` method that our `Trainer` did reload the
best model properly (if it was not the last one)."""

trainer.evaluate()

# Hyperparameter search


def model_init():
    return AutoModelForSequenceClassification.from_pretrained(
        model_checkpoint, num_labels=num_labels
    )


trainer = Trainer(
    model_init=model_init,
    args=args,
    train_dataset=encoded_dataset["train"],
    eval_dataset=encoded_dataset[validation_key],
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)

"""You can try to find some good hyperparameter on a portion of the training
dataset by replacing the `train_dataset` line above by:

train_dataset = encoded_dataset["train"].shard(index=1, num_shards=10)

for 1/10th of the dataset.
"""

best_run = trainer.hyperparameter_search(n_trials=10, direction="maximize")

"""The `hyperparameter_search` method returns a `BestRun` objects, which
contains the value of the objective maximized (by default the sum of all
metrics) and the hyperparameters it used for that run.

You can customize the objective to maximize by passing along a
`compute_objective` function to the `hyperparameter_search` method, and you can
customize the search space by passing a `hp_space` argument to
`hyperparameter_search`. See this [forum
post](https://discuss.huggingface.co/t/using-hyperparameter-search-in-trainer/785/10)
for some examples.
"""

for n, v in best_run.hyperparameters.items():
    setattr(trainer.args, n, v)

trainer.train()

"""Don't forget to [update your
model](https://huggingface.co/transformers/model_sharing.html) on the [🤗 Model
Hub](https://huggingface.co/models). You can then use it only to generate
results like the one shown in the first picture of this notebook!"""
