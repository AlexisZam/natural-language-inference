dataset_info_dict = {
    "anli_r1": {
        "path": "anli",
        "name": None,
        "text": "premise",
        "text_pair": "hypothesis",
        "train_dataset": "train_r1",
        "eval_dataset": "dev_r1",
        "test_dataset": "test_r1",
    },
    "anli_r2": {
        "path": "anli",
        "name": None,
        "text": "premise",
        "text_pair": "hypothesis",
        "train_dataset": "train_r2",
        "eval_dataset": "dev_r2",
        "test_dataset": "test_r2",
    },
    "anli_r3": {
        "path": "anli",
        "name": None,
        "text": "premise",
        "text_pair": "hypothesis",
        "train_dataset": "train_r3",
        "eval_dataset": "dev_r3",
        "test_dataset": "test_r3",
    },
    "mnli_matched": {
        "path": "glue",
        "name": "mnli",
        "text": "premise",
        "text_pair": "hypothesis",
        "train_dataset": "train",
        "eval_dataset": "validation_matched",
        "test_dataset": None,
    },
    "mnli_mismatched": {
        "path": "glue",
        "name": "mnli",
        "text": "premise",
        "text_pair": "hypothesis",
        "train_dataset": "train",
        "eval_dataset": "validation_mismatched",
        "test_dataset": None,
    },
    "qnli": {
        "path": "glue",
        "name": "qnli",
        "text": "question",
        "text_pair": "sentence",
        "train_dataset": "train",
        "eval_dataset": "validation",
        "test_dataset": None,
    },
    "rte": {
        "path": "glue",
        "name": "rte",
        "text": "sentence1",
        "text_pair": "sentence2",
        "train_dataset": "train",
        "eval_dataset": "validation",
        "test_dataset": None,
    },
    "scitail": {
        "path": "scitail",
        "name": "tsv_format",
        "text": "premise",
        "text_pair": "hypothesis",
        "train_dataset": "train",
        "eval_dataset": "validation",
        "test_dataset": "test",
    },
    "snli": {
        "path": "snli",
        "name": None,
        "text": "premise",
        "text_pair": "hypothesis",
        "train_dataset": "train",
        "eval_dataset": "validation",
        "test_dataset": "test",
    },
    "wnli": {
        "path": "glue",
        "name": "wnli",
        "text": "sentence1",
        "text_pair": "sentence2",
        "train_dataset": "train",
        "eval_dataset": "validation",
        "test_dataset": None,
    },
}
