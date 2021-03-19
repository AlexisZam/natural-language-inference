from dataclasses import dataclass, field

from dataset_info_dict import dataset_info_dict
from transformers import TrainingArguments


@dataclass
class DatasetArguments:
    dataset_name: str.lower = field(
        metadata={
            "choices": dataset_info_dict.keys(),
            "help": "The name of the dataset to train on.",  # FIXME
        },
    )
    max_length: int = field(
        default=128,
        metadata={
            "help": "The maximum total input sequence length after tokenization. Sequences longer than this will be truncated, sequences shorter will be padded."
        },
    )  # FIXME
    padding: str = field(
        default="max_length",
        metadata={
            "choices": ("do_not_pad", "max_length"),
            "help": "Whether to pad all samples to `max_length`. If do_not_pad, will pad the samples dynamically when batching to the maximum length in the batch.",
        },
    )  # FIXME


@dataclass
class ModelArguments:
    pretrained_model_name: str = field(
        metadata={
            "help": "A string, the model id of a pretrained model configuration hosted inside a model repo on huggingface.co. Valid model ids can be located at the root-level, like bert-base-uncased, or namespaced under a user or organization name, like dbmdz/bert-base-german-cased."
        }
    )


@dataclass
class MyTrainingArguments(TrainingArguments):
    do_hyperparameter_search: bool = field(
        default=False,
        metadata={
            "help": "Whether to run hyperparameter search or not. This argument is not directly used by :class:`~transformers.Trainer`, it's intended to be used by your training/evaluation scripts instead."
        },
    )
    n_trials: int = field(
        default=20, metadata={"help": "The number of trial runs to test."}
    )
