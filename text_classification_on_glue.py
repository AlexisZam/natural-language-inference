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

"""A 🤗 Transformers `Tokenizer` will tokenize the inputs (including converting
the tokens to their corresponding IDs in the pretrained vocabulary) and put it
in a format the model expects, as well as generate the other inputs that model
requires.

To do all of this, we instantiate our tokenizer with the
`AutoTokenizer.from_pretrained` method, which will ensure:

- we get a tokenizer that corresponds to the model architecture we want to use,
- we download the vocabulary used when pretraining this specific checkpoint.
"""

"""We just feed our samples to the `tokenizer` with the argument
`truncation=True`. This will ensure that an input longer that what the model
selected can handle will be truncated to the maximum length accepted by the
model."""

"""The warning is telling us we are throwing away some weights (the
`vocab_transform` and `vocab_layer_norm` layers) and randomly initializing some
other (the `pre_classifier` and `classifier` layers). This is absolutely normal
in this case, because we are removing the head used to pretrain the model on a
masked language modeling objective and replacing it with a new head for which we
don't have pretrained weights, so the library warns us we should fine-tune this
model before using it for inference, which is exactly what we are going to do.
"""

"""The only preprocessing we have to do is to take the argmax of our predicted
logits.
"""

"""You might wonder why we pass along the `tokenizer` when we already
preprocessed our data. This is because we will use it once last time to make all
the samples we gather the same length by applying padding, which requires
knowing the model's preferences regarding padding (to the left or right? with
which token?). The `tokenizer` has a pad method that will do all of this right
for us, and the `Trainer` will use it. You can customize this part by defining
and passing your own `data_collator` which will receive the samples like the
dictionaries seen above and will need to return a dictionary of tensors.
"""

"""We can check with the `evaluate` method that our `Trainer` did reload the
best model properly (if it was not the last one)."""

"""You can try to find some good hyperparameter on a portion of the training
dataset by replacing the `train_dataset` line above by:

train_dataset = encoded_dataset["train"].shard(index=1, num_shards=10)

for 1/10th of the dataset.
"""

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

"""Don't forget to [update your
model](https://huggingface.co/transformers/model_sharing.html) on the [🤗 Model
Hub](https://huggingface.co/models). You can then use it only to generate
results like the one shown in the first picture of this notebook!"""
