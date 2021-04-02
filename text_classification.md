We pass along the `tokenizer` because we will use it once last time to make all
the samples we gather the same length by applying padding, which requires
knowing the model's preferences regarding padding (to the left or right? with
which token?). You can customize this part by defining and passing your own
`data_collator` which will receive the samples like the dictionaries seen above
and will need to return a dictionary of tensors.

During hyperparameter search, the `Trainer` will run several trainings, so it
needs to have the model defined via a function (so it can be reinitialized at
each new run) instead of just having it passed.
