from datasets import ClassLabel, load_dataset

from dataset_info_dict import dataset_info_dict


def my_load_dataset(dataset_arguments, tokenizer):
    dataset_info = dataset_info_dict[dataset_arguments.dataset_name]

    # FIXME
    if dataset_arguments.max_length > tokenizer.model_max_length:
        print(
            f"WARNING:{__name__}:The max_length passed ({dataset_arguments.max_length}) is larger than the maximum length for the model ({tokenizer.model_max_length}).",
            f"Using max_length={tokenizer.model_max_length}.",
        )

    function = lambda batch: tokenizer(
        batch[dataset_info["text"]],
        text_pair=batch[dataset_info["text_pair"]],
        padding=dataset_arguments.padding,
        truncation=True,
        max_length=min(dataset_arguments.max_length, tokenizer.model_max_length),
    )  # FIXME

    dataset_dict = (
        load_dataset(dataset_info["path"], name=dataset_info["name"])
        .filter(lambda example: example["label"] != -1)
        .map(function, batched=True)
    )

    # FIXME
    if dataset_arguments.dataset_name == "scitail":
        label = ClassLabel(names=("entails", "neutral"))

        def function(batch):
            batch["label"] = label.str2int(batch["label"])
            return batch

        dataset_dict = dataset_dict.map(function, batched=True)

    # FIXME
    return {
        "train": dataset_dict[dataset_info["train"]],
        "eval": dataset_dict[dataset_info["eval"]],
        "test": None
        if dataset_info["test"] is None
        else dataset_dict[dataset_info["test"]],
    }
