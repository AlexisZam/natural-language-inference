Depending on the model you selected, you will see different keys in the
dictionary returned by the cell above. They don't matter much for what we're
doing here (just know they are required by the model we will instantiate later),
you can learn more about them in [this
tutorial](https://huggingface.co/transformers/preprocessing.html) if you're
interested.

The warning is telling us we are throwing away some weights (the
`vocab_transform` and `vocab_layer_norm` layers) and randomly initializing some
other (the `pre_classifier` and `classifier` layers). This is absolutely normal
in this case, because we are removing the head used to pretrain the model on a
masked language modeling objective and replacing it with a new head for which we
don't have pretrained weights, so the library warns us we should fine-tune this
model before using it for inference, which is exactly what we are going to do.

We pass along the `tokenizer` because we will use it once last time to make all
the samples we gather the same length by applying padding, which requires
knowing the model's preferences regarding padding (to the left or right? with
which token?). The `tokenizer` has a pad method that will do all of this right
for us, and the `Trainer` will use it. You can customize this part by defining
and passing your own `data_collator` which will receive the samples like the
dictionaries seen above and will need to return a dictionary of tensors.

During hyperparameter search, the `Trainer` will run several trainings, so it
needs to have the model defined via a function (so it can be reinitialized at
each new run) instead of just having it passed.

Then you can run a full training on the best hyperparameters picked by the
search.

Don't forget to [upload your
model](https://huggingface.co/transformers/model_sharing.html) on the [🤗 Model
Hub](https://huggingface.co/models). You can then use it only to generate
results like the one shown in the first picture of this notebook!
