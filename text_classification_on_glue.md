[QNLI](https://rajpurkar.github.io/SQuAD-explorer/) (Question-answering Natural
Language Inference) Determine if the answer to a question is in the second
sentence or not. (This dataset is built from the SQuAD dataset.)

[WNLI](https://cs.nyu.edu/faculty/davise/papers/WinogradSchemas/WS.html)
(Winograd Natural Language Inference) Determine if a sentence with an anonymous
pronoun and a sentence with this pronoun replaced are entailed or not. (This
dataset is built from the Winograd Schema Challenge dataset.)

This notebook is built to run with any model checkpoint from the [Model
Hub](https://huggingface.co/models) as long as that model has a version with a
classification head.

The `AutoTokenizer.from_pretrained` method will ensure we download the
vocabulary used when pretraining this specific checkpoint.

The warning is telling us we are throwing away some weights (the
`vocab_transform` and `vocab_layer_norm` layers) and randomly initializing some
other (the `pre_classifier` and `classifier` layers). This is absolutely normal
in this case, because we are removing the head used to pretrain the model on a
masked language modeling objective and replacing it with a new head for which we
don't have pretrained weights, so the library warns us we should fine-tune this
model before using it for inference, which is exactly what we are going to do.

The only preprocessing we have to do is to take the argmax of our predicted
logits.

We pass along the `tokenizer` because we will use it once last time to make all
the samples we gather the same length by applying padding, which requires
knowing the model's preferences regarding padding (to the left or right? with
which token?). The `tokenizer` has a pad method that will do all of this right
for us, and the `Trainer` will use it. You can customize this part by defining
and passing your own `data_collator` which will receive the samples like the
dictionaries seen above and will need to return a dictionary of tensors.

```python
train_dataset = train_dataset.shard(index=1, num_shards=10)
```

You can customize the objective to maximize by passing along a
`compute_objective` function to the `hyperparameter_search` method. See this
[forum
post](https://discuss.huggingface.co/t/using-hyperparameter-search-in-trainer/785/10)
for some examples.

Don't forget to [update your
model](https://huggingface.co/transformers/model_sharing.html) on the [🤗 Model
Hub](https://huggingface.co/models). You can then use it only to generate
results like the one shown in the first picture of this notebook!
