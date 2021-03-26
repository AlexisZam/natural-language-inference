# Natural Language Inference

## Requirements

- [Ubuntu](https://ubuntu.com/)

- [Python](https://www.python.org/)
- [Datasets](https://github.com/huggingface/datasets)
- [Transformers](https://github.com/huggingface/transformers)
- [PyTorch](https://pytorch.org/)
- [Optuna](https://optuna.org/)
- [scikit-learn](https://scikit-learn.org/)

- [CUDA Toolkit](https://developer.nvidia.com/cuda-toolkit)
- [NVIDIA cuDNN](https://developer.nvidia.com/cudnn)

## Datasets

| Dataset                                                                               | Website                                | Paper                                                                              | Code                                       | Hugging Face                                                                                 | Misc                        |
| ------------------------------------------------------------------------------------- | -------------------------------------- | ---------------------------------------------------------------------------------- | ------------------------------------------ | -------------------------------------------------------------------------------------------- | --------------------------- |
| A large annotated corpus for learning natural language inference                      | https://nlp.stanford.edu/projects/snli | https://arxiv.org/abs/1508.05326                                                   |                                            | https://huggingface.co/datasets/snli                                                         |                             |
| A Broad-Coverage Challenge Corpus for Sentence Understanding through Inference        | https://cims.nyu.edu/~sbowman/multinli | https://arxiv.org/abs/1704.05426                                                   | https://github.com/nyu-mll/multiNLI        | https://huggingface.co/datasets/multi_nli https://huggingface.co/datasets/multi_nli_mismatch |                             |
| GLUE: A Multi-Task Benchmark and Analysis Platform for Natural Language Understanding | https://gluebenchmark.com/             | https://arxiv.org/abs/1804.07461                                                   | https://github.com/nyu-mll/GLUE-baselines  | https://huggingface.co/datasets/glue                                                         |                             |
| SciTaiL: A Textual Entailment Dataset from Science Question Answering                 | https://allenai.org/data/scitail       | http://ai2-website.s3.amazonaws.com/publications/scitail-aaai-2018_cameraready.pdf | https://github.com/allenai/scitail         | https://huggingface.co/datasets/scitail                                                      |                             |
| e-SNLI: Natural Language Inference with Natural Language Explanations                 |                                        | https://arxiv.org/abs/1812.01193                                                   | https://github.com/OanaMariaCamburu/e-SNLI | https://huggingface.co/datasets/esnli                                                        |                             |
| Adversarial NLI: A New Benchmark for Natural Language Understanding                   |                                        | https://arxiv.org/abs/1910.14599                                                   | https://github.com/facebookresearch/anli   | https://huggingface.co/datasets/anli                                                         | https://adversarialnli.com/ |

## Models

| Model                                                                            | Paper                            | Code                                                                                                 | Hugging Face                                                                                |
| -------------------------------------------------------------------------------- | -------------------------------- | ---------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------- |
| BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding | https://arxiv.org/abs/1810.04805 | https://github.com/google-research/bert                                                              | https://huggingface.co/bert-base-uncased https://huggingface.co/bert-base-cased             |
| DistilBERT, a distilled version of BERT: smaller, faster, cheaper and lighter    | https://arxiv.org/abs/1910.01108 | https://github.com/huggingface/transformers https://github.com/huggingface/swift-coreml-transformers | https://huggingface.co/distilbert-base-uncased https://huggingface.co/distilbert-base-cased |
