# Text classification examples

## PyTorch version

We get the following results on the dev set of the benchmark with the previous
commands (with an exception for WNLI which is tiny and where we used 5 epochs
isntead of 3). Trainings times are given for information (a single Titan RTX was
used). Using mixed precision training usually results in 2x-speedup for training
with the same final results.

| Task | Metric                       | Result      | Training time | Result (FP16) | Training time (FP16) |
| ---- | ---------------------------- | ----------- | ------------- | ------------- | -------------------- |
| MNLI | Matched acc./Mismatched acc. | 83.91/84.10 | 2:35:23       | 84.04/84.06   | 1:17:06              |
| QNLI | Accuracy                     | 90.66       | 40:57         | 90.96         | 20:16                |
| RTE  | Accuracy                     | 65.70       | 57            | 65.34         | 29                   |
| WNLI | Accuracy                     | 56.34       | 24            | 56.34         | 12                   |

Some of these results are significantly different from the ones reported on the
test set of GLUE benchmark on the website.

I get weird results for WNLI. What gives? The train/dev split for WNLI is
correct, but turns out to be somewhat adversarial: when two examples contain the
same sentence, that usually means they'll have opposite labels. The train and
dev splits may share sentences, so if a model has overfit the training set, it
may get worse than chance accuracy on WNLI on the dev set. Additionally, the
test set has a different label distribution than the train and dev sets.

## PyTorch version, no Trainer

Like `run_glue.py`, this script allows you to fine-tune any of the models on the
[hub](https://huggingface.co/models) on a text classification task, either a
GLUE task or your own data in a csv or a JSON file. The main difference is that
this script exposes the bare training loop, to allow you to quickly experiment
and add any customization you would like.

It offers less options than the script with `Trainer` (for instance you can
easily change the options for the optimizer or the dataloaders directly in the
script) but still run in a distributed setup, on TPU and supports mixed
precision by the mean of the [🤗
`Accelerate`](https://github.com/huggingface/accelerate) library. You can use
the script normally after installing it:

```bash
pip install accelerate
```

then

```bash
export TASK_NAME=mrpc

python run_glue_no_trainer.py \
  --model_name_or_path bert-base-cased \
  --task_name $TASK_NAME \
  --max_length 128 \
  --per_device_train_batch_size 32 \
  --learning_rate 2e-5 \
  --num_train_epochs 3 \
  --output_dir /tmp/$TASK_NAME/
```

You can then use your usual launchers to run in it in a distributed environment,
but the easiest way is to run

```bash
accelerate config
```

and reply to the questions asked. Then

```bash
accelerate test
```

that will check everything is ready for training. Finally, you cna launch
training with

```bash
export TASK_NAME=mrpc

accelerate launch run_glue_no_trainer.py \
  --model_name_or_path bert-base-cased \
  --task_name $TASK_NAME \
  --max_length 128 \
  --per_device_train_batch_size 32 \
  --learning_rate 2e-5 \
  --num_train_epochs 3 \
  --output_dir /tmp/$TASK_NAME/
```
