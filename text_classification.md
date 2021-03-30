In this notebook, we will see how to fine-tune one of the [🤗
Transformers](https://github.com/huggingface/transformers) model to a text
classification task of the [GLUE Benchmark](https://gluebenchmark.com/).

The GLUE Benchmark is a group of nine classification tasks on sentences or pairs
of sentences which are:

- MNLI (Multi-Genre Natural Language Inference) Determine if a sentence entails,
  contradicts or is unrelated to a given hypothesis. (This dataset has two
  versions, one with the validation and test set coming from the same
  distribution, another where the validation and test use out-of-domain data.)
- QNLI (Question-answering Natural Language Inference) Determine if the answer
  to a question is in the second sentence or not. (This dataset is built from
  the SQuAD dataset.)
- RTE (Recognizing Textual Entailment) Determine if a sentence entails a given
  hypothesis or not.
- WNLI (Winograd Natural Language Inference) Determine if a sentence with an
  anonymous pronoun and a sentence with this pronoun replaced are entailed or
  not. (This dataset is built from the Winograd Schema Challenge dataset.)

This notebook is built to run with any model checkpoint from the [Model
Hub](https://huggingface.co/models) as long as that model has a version with a
classification head. Depending on you model and the GPU you are using, you might
need to adjust the batch size to avoid out-of-memory errors.

## Preprocessing the data

This is done by a 🤗 Transformers `Tokenizer` which will (as the name indicates)
tokenize the inputs (including converting the tokens to their corresponding IDs
in the pretrained vocabulary) and put it in a format the model expects, as well
as generate the other inputs that model requires.

To do all of this, we instantiate our tokenizer with the
`AutoTokenizer.from_pretrained` method, which will ensure:

- we get a tokenizer that corresponds to the model architecture we want to use,
- we download the vocabulary used when pretraining this specific checkpoint.

Depending on the model you selected, you will see different keys in the
dictionary returned by the cell above. They don't matter much for what we're
doing here (just know they are required by the model we will instantiate later),
you can learn more about them in [this
tutorial](https://huggingface.co/transformers/preprocessing.html) if you're
interested.

We just feed them to the `tokenizer` with the argument `truncation=True`. This
will ensure that an input longer that what the model selected can handle will be
truncated to the maximum length accepted by the model.

In the case of several examples, the tokenizer will return a list of lists for
each key.

Note that we passed `batched=True` to encode the texts by batches together. This
is to leverage the full benefit of the fast tokenizer we loaded earlier, which
will use multi-threading to treat the texts in a batch concurrently.

## Fine-tuning the model

Now that our data is ready, we can download the pretrained model and fine-tune
it. Since all our tasks are about sentence classification, we use the
`AutoModelForSequenceClassification` class. Like with the tokenizer, the
`from_pretrained` method will download and cache the model for us. The only
thing we have to specify is the number of labels for our problem (which is
always 2, except for STS-B which is a regression problem and MNLI where we have
3 labels).

The warning is telling us we are throwing away some weights (the
`vocab_transform` and `vocab_layer_norm` layers) and randomly initializing some
other (the `pre_classifier` and `classifier` layers). This is absolutely normal
in this case, because we are removing the head used to pretrain the model on a
masked language modeling objective and replacing it with a new head for which we
don't have pretrained weights, so the library warns us we should fine-tune this
model before using it for inference, which is exactly what we are going to do.

It requires one folder name, which will be used to save the checkpoints of the
model.

Here we set the evaluation to be done at the end of each epoch, tweak the
learning rate, use the `batch_size` defined at the top of the notebook and
customize the number of epochs for training, as well as the weight decay. Since
the best model might not be the one at the end of training, we ask the `Trainer`
to load the best model it saved (according to `metric_name`) at the end of
training.

The only preprocessing we have to do is to take the argmax of our predicted
logits.

We pass along the `tokenizer` because we will use it once last time to make all
the samples we gather the same length by applying padding, which requires
knowing the model's preferences regarding padding (to the left or right? with
which token?). The `tokenizer` has a pad method that will do all of this right
for us, and the `Trainer` will use it. You can customize this part by defining
and passing your own `data_collator` which will receive the samples like the
dictionaries seen above and will need to return a dictionary of tensors.

We can check with the `evaluate` method that our `Trainer` did reload the best
model properly (if it was not the last one).

To see how your model fared you can compare it to the [GLUE Benchmark
leaderboard](https://gluebenchmark.com/leaderboard).

## Hyperparameter search

During hyperparameter search, the `Trainer` will run several trainings, so it
needs to have the model defined via a function (so it can be reinitialized at
each new run) instead of just having it passed.

The method we call this time is `hyperparameter_search`. Note that it can take a
long time to run on the full dataset for some of the tasks. You can try to find
some good hyperparameter on a portion of the training dataset by replacing the
`train_dataset` line above by:

```python
train_dataset = encoded_dataset["train"].shard(index=1, num_shards=10)
```

for 1/10th of the dataset. Then you can run a full training on the best
hyperparameters picked by the search.

The `hyperparameter_search` method returns a `BestRun` objects, which contains
the value of the objective maximized (by default the sum of all metrics) and the
hyperparameters it used for that run.

You can customize the objective to maximize by passing along a
`compute_objective` function to the `hyperparameter_search` method, and you can
customize the search space by passing a `hp_space` argument to
`hyperparameter_search`. See this [forum
post](https://discuss.huggingface.co/t/using-hyperparameter-search-in-trainer/785/10)
for some examples.

Don't forget to [upload your
model](https://huggingface.co/transformers/model_sharing.html) on the [🤗 Model
Hub](https://huggingface.co/models). You can then use it only to generate
results like the one shown in the first picture of this notebook!

```python
model = AutoModelForSequenceClassification.from_pretrained(
    model_checkpoint, num_labels=num_labels
)


def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = predictions[:, 0]
    return metric.compute(predictions=predictions, references=labels)
```
