from dataclasses import dataclass, field
from typing import Optional

task_names = (
    "anli_r1",
    "anli_r2",
    "anli_r3",
    "mnli",
    "qnli",
    "rte",
    "scitail",
    "snli",
    "wnli",
)


@dataclass
class DatasetArguments:
    """
    Arguments pertaining to what data we are going to input our model for
    training and eval.

    Using `HfArgumentParser` we can turn this class into argparse arguments to
    be able to specify them on the command line.
    """

    task_name: str.lower = field(
        metadata={
            "choices": task_names,
            "help": "The name of the task to train on: " + ", ".join(task_names),
        },
    )
    max_seq_length: int = field(
        default=128,
        metadata={
            "help": "The maximum total input sequence length after tokenization. Sequences longer than this will be truncated, sequences shorter will be padded."
        },
    )
    load_from_cache_file: bool = field(
        default=True,
        metadata={
            "help": "If a cache file storing the current computation from function can be identified, use it instead of recomputing."
        },
    )
    pad_to_max_length: bool = field(
        default=True,
        metadata={
            "help": "Whether to pad all samples to `max_seq_length`. If False, will pad the samples dynamically when batching to the maximum length in the batch."
        },
    )


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to
    fine-tune from.
    """

    model_name_or_path: str = field(
        metadata={
            "help": "Path to pretrained model or model identifier from huggingface.co/models"
        }
    )
    config_name: Optional[str] = field(
        default=None,
        metadata={
            "help": "Pretrained config name or path if not the same as model_name"
        },
    )
    tokenizer_name: Optional[str] = field(
        default=None,
        metadata={
            "help": "Pretrained tokenizer name or path if not the same as model_name"
        },
    )
    use_fast: bool = field(
        default=True,
        metadata={
            "help": "Whether or not to try to load the fast version of the tokenizer."
        },
    )
    revision: str = field(
        default="main",
        metadata={
            "help": "The specific model version to use. It can be a branch name, a tag name, or a commit id, since we use a git-based system for storing models and other artifacts on huggingface.co, so revision can be any identifier allowed by git."
        },
    )
    use_auth_token: bool = field(
        default=False,
        metadata={
            "help": "Will use the token generated when running `transformers-cli login` (necessary to use this script with private models)."
        },
    )
