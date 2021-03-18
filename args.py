from dataclasses import dataclass, field

from transformers import TrainingArguments

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
    task_name: str.lower = field(
        metadata={
            "choices": task_names,
            "help": "The name of the task to train on: " + ", ".join(task_names),
        },
    )
    max_length: int = field(
        default=128,
        metadata={
            "help": "The maximum total input sequence length after tokenization. Sequences longer than this will be truncated, sequences shorter will be padded."
        },
    )
    pad_to_max_length: bool = field(
        default=True,
        metadata={
            "help": "Whether to pad all samples to `max_length`. If False, will pad the samples dynamically when batching to the maximum length in the batch."
        },
    )


@dataclass
class ModelArguments:
    model_name_or_path: str = field(
        metadata={
            "help": "Path to pretrained model or model identifier from huggingface.co/models"
        }
    )


@dataclass
class MyTrainingArguments(TrainingArguments):
    do_hyperparameter_search: bool = field(
        default=False, metadata={"help": "Whether to run hyperparameter search."}
    )
