from dataclasses import dataclass
from typing import Optional

from datasets import ClassLabel, Features, Value, load_dataset, load_metric


@dataclass
class DatasetInfo:
    path: str
    name: Optional[str] = None
    train: str = "train"
    validation: str = "validation"
    test: Optional[str] = "test"
    text: str = "premise"
    text_pair: str = "hypothesis"
    features: Optional[Features] = None


dataset_infos = {
    "anli_r1": DatasetInfo(
        "anli", train="train_r1", validation="dev_r1", test="test_r1"
    ),
    "anli_r2": DatasetInfo(
        "anli", train="train_r2", validation="dev_r2", test="test_r2"
    ),
    "anli_r3": DatasetInfo(
        "anli", train="train_r3", validation="dev_r3", test="test_r3"
    ),
    "esnli": DatasetInfo("esnli"),
    "glue/qnli": DatasetInfo(
        "glue", name="qnli", test=None, text="question", text_pair="sentence"
    ),
    "glue/rte": DatasetInfo(
        "glue", name="rte", test=None, text="sentence1", text_pair="sentence2"
    ),
    "glue/wnli": DatasetInfo(
        "glue", name="wnli", test=None, text="sentence1", text_pair="sentence2"
    ),
    "multi_nli_matched": DatasetInfo(
        "multi_nli", validation="validation_matched", test=None
    ),
    "multi_nli_mismatched": DatasetInfo(
        "multi_nli", validation="validation_mismatched", test=None
    ),
    "scitail": DatasetInfo(
        "scitail",
        name="tsv_format",
        features=Features(
            {
                "premise": Value("string"),
                "hypothesis": Value("string"),
                "label": ClassLabel(names=("entails", "neutral")),
            }
        ),
    ),
    "snli": DatasetInfo("snli"),
    "super_glue/cb": DatasetInfo("super_glue", name="cb", test=None),  # TODO f1
    "super_glue/rte": DatasetInfo("super_glue", name="rte", test=None),
}
