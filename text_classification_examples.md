We get the following results on the dev set of the benchmark with the previous
commands (with an exception for WNLI which is tiny and where we used 5 epochs
isntead of 3).

Some of these results are significantly different from the ones reported on the
test set of GLUE benchmark on the website. For WNLI, please refer to [FAQ
#12](https://gluebenchmark.com/faq) on the website.

| Task | Metric                       | Result      | Training time | Result (FP16) | Training time (FP16) |
| ---- | ---------------------------- | ----------- | ------------- | ------------- | -------------------- |
| MNLI | Matched acc./Mismatched acc. | 83.91/84.10 | 2:35:23       | 84.04/84.06   | 1:17:06              |
| QNLI | Accuracy                     | 90.66       | 40:57         | 90.96         | 20:16                |
| RTE  | Accuracy                     | 65.70       | 57            | 65.34         | 29                   |
| WNLI | Accuracy                     | 56.34       | 24            | 56.34         | 12                   |
