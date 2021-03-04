# Text classification examples

## PyTorch version

We get the following results on the dev set of the benchmark with the previous commands (with an exception for
WNLI which is tiny and where we used 5 epochs isntead of 3). Trainings are seeded so you should obtain the same
results with PyTorch 1.6.0 (and close results with different versions), training times are given for information (a
single Titan RTX was used).

Some of these results are significantly different from the ones reported on the test set of GLUE benchmark on the
website. For WNLI, please refer to [FAQ #12](https://gluebenchmark.com/faq) on the website.

Using mixed precision training usually results in 2x-speedup for training with the same final results.

| Task | Metric                       | Result      | Training time | Result (FP16) | Training time (FP16) |
| ---- | ---------------------------- | ----------- | ------------- | ------------- | -------------------- |
| MNLI | Matched acc./Mismatched acc. | 83.91/84.10 | 2:35:23       | 84.04/84.06   | 1:17:06              |
| QNLI | Accuracy                     | 90.66       | 40:57         | 90.96         | 20:16                |
| RTE  | Accuracy                     | 65.70       | 57            | 65.34         | 29                   |
| WNLI | Accuracy                     | 56.34       | 24            | 56.34         | 12                   |

# Run TensorFlow 2.0 version

Based on the script [`run_tf_glue.py`](https://github.com/huggingface/transformers/blob/master/examples/text-classification/run_tf_glue.py).

Fine-tuning the library TensorFlow 2.0 Bert model for sequence classification on the MRPC task of the GLUE benchmark: [General Language Understanding Evaluation](https://gluebenchmark.com/).

This script has an option for mixed precision (Automatic Mixed Precision / AMP) to run models on Tensor Cores (NVIDIA Volta/Turing GPUs) and future hardware and an option for XLA, which uses the XLA compiler to reduce model runtime.
Options are toggled using `USE_XLA` or `USE_AMP` variables in the script.
These options and the below benchmark are provided by @tlkh.

Quick benchmarks from the script (no other modifications):

| GPU     | Mode | Time (2nd epoch) | Val Acc (3 runs)     |
| ------- | ---- | ---------------- | -------------------- |
| Titan V | FP32 | 41s              | 0.8438/0.8281/0.8333 |
| Titan V | AMP  | 26s              | 0.8281/0.8568/0.8411 |
| V100    | FP32 | 35s              | 0.8646/0.8359/0.8464 |
| V100    | AMP  | 22s              | 0.8646/0.8385/0.8411 |
| 1080 Ti | FP32 | 55s              | -                    |

Mixed precision (AMP) reduces the training time considerably for the same hardware and hyper-parameters (same batch size was used).
