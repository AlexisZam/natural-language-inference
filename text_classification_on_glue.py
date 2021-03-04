#!/usr/bin/env python

import numpy as np
from datasets import load_dataset, load_metric

from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)

"""Original file is located at
    https://colab.research.google.com/github/huggingface/notebooks/blob/master/examples/text_classification.ipynb

- [MNLI](https://arxiv.org/abs/1704.05426) (Multi-Genre Natural Language
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
should run smoothly:"""

task = "rte"
model_checkpoint = "distilbert-base-uncased"
batch_size = 16

# Loading the dataset

actual_task = "mnli" if task == "mnli-mm" else task
dataset = load_dataset("glue", actual_task)
metric = load_metric("accuracy")

# Preprocessing the data

"""Before we can feed those texts to our model, we need to preprocess them. This is
done by a 🤗 Transformers `Tokenizer` which will (as the name indicates)
tokenize the inputs (including converting the tokens to their corresponding IDs
in the pretrained vocabulary) and put it in a format the model expects, as well
as generate the other inputs that model requires.

To do all of this, we instantiate our tokenizer with the
`AutoTokenizer.from_pretrained` method, which will ensure:

- we get a tokenizer that corresponds to the model architecture we want to use,
- we download the vocabulary used when pretraining this specific checkpoint.
"""

tokenizer = AutoTokenizer.from_pretrained(model_checkpoint, use_fast=True)

"""We pass along `use_fast=True` to the call above to use one of the fast
tokenizers (backed by Rust) from the 🤗 Tokenizers library. Those fast
tokenizers are available for almost all models, but if you got an error with the
previous call, remove that argument.
"""

task_to_keys = {
    "mnli": ("premise", "hypothesis"),
    "mnli-mm": ("premise", "hypothesis"),
    "qnli": ("question", "sentence"),
    "rte": ("sentence1", "sentence2"),
    "wnli": ("sentence1", "sentence2"),
}

sentence1_key, sentence2_key = task_to_keys[task]

"""We can them write the function that will preprocess our samples. We just feed
them to the `tokenizer` with the argument `truncation=True`. This will ensure
that an input longer that what the model selected can handle will be truncated
to the maximum length accepted by the model."""


def preprocess_function(examples):
    return tokenizer(examples[sentence1_key], examples[sentence2_key], truncation=True)


"""This function works with one or several examples. In the case of several
examples, the tokenizer will return a list of lists for each key:"""

encoded_dataset = dataset.map(preprocess_function, batched=True)

"""Even better, the results are automatically cached by the 🤗 Datasets library
to avoid spending time on this step the next time you run your notebook. The 🤗
Datasets library is normally smart enough to detect when the function you pass
to map has changed (and thus requires to not use the cache data). For instance,
it will properly detect if you change the task in the first cell and rerun the
notebook. 🤗 Datasets warns you when it uses cached files, you can pass
`load_from_cache_file=False` in the call to `map` to not use the cached files
and force the preprocessing to be applied again.

Note that we passed `batched=True` to encode the texts by batches together. This
is to leverage the full benefit of the fast tokenizer we loaded earlier, which
will use multi-threading to treat the texts in a batch concurrently."""

# Fine-tuning the model

"""Now that our data is ready, we can download the pretrained model and fine-tune
it. Since all our tasks are about sentence classification, we use the
`AutoModelForSequenceClassification` class. Like with the tokenizer, the
`from_pretrained` method will download and cache the model for us. The only
thing we have to specify is the number of labels for our problem (which is
always 2, except for STS-B which is a regression problem and MNLI where we have
3 labels):
"""

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

To instantiate a `Trainer`, we will need to define two more things. The most
important is the
[`TrainingArguments`](https://huggingface.co/transformers/main_classes/trainer.html#transformers.TrainingArguments),
which is a class that contains all the attributes to customize the training. It
requires one folder name, which will be used to save the checkpoints of the
model, and all other arguments are optional:
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

"""Here we set the evaluation to be done at the end of each epoch, tweak the
learning rate, use the `batch_size` defined at the top of the notebook and
customize the number of epochs for training, as well as the weight decay. Since
the best model might not be the one at the end of training, we ask the `Trainer`
to load the best model it saved (according to `metric_name`) at the end of
training.

The last thing to define for our `Trainer` is how to compute the metrics from
the predictions. We need to define a function for this, which will just use the
`metric` we loaded earlier, the only preprocessing we have to do is to take the
argmax of our predicted logits (our just squeeze the last axis in the case of
STS-B):
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
best model properly (if it was not the last one):"""

trainer.evaluate()

"""To see how your model fared you can compare it to the [GLUE Benchmark
leaderboard](https://gluebenchmark.com/leaderboard)."""

# Hyperparameter search


def model_init():
    return AutoModelForSequenceClassification.from_pretrained(
        model_checkpoint, num_labels=num_labels
    )


"""And we can instantiate our `Trainer` like before:"""

trainer = Trainer(
    model_init=model_init,
    args=args,
    train_dataset=encoded_dataset["train"],
    eval_dataset=encoded_dataset[validation_key],
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)

"""The method we call this time is `hyperparameter_search`. Note that it can
take a long time to run on the full dataset for some of the tasks. You can try
to find some good hyperparameter on a portion of the training dataset by
replacing the `train_dataset` line above by:
```python
train_dataset = encoded_dataset["train"].shard(index=1, num_shards=10) 
```
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
