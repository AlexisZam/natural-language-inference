from datasets import ClassLabel, load_dataset

from dataset_info_dict import dataset_info_dict


def my_load_dataset(dataset_arguments, tokenizer):
    dataset_info = dataset_info_dict[dataset_arguments.dataset_name]

    function = lambda batch: tokenizer(
        batch[dataset_info["text"]],
        text_pair=batch[dataset_info["text_pair"]],
        padding="max_length",
        truncation=True,
        max_length=min(128, tokenizer.model_max_length),
    )

    dataset_dict = (
        load_dataset(dataset_info["path"], name=dataset_info["name"])
        .filter(lambda example: example["label"] != -1)
        .map(function, batched=True)
    )

    if dataset_arguments.dataset_name == "scitail":
        class_label = ClassLabel(names=("entails", "neutral"))

        def function(batch):
            batch["label"] = class_label.str2int(batch["label"])
            return batch

        dataset_dict = dataset_dict.map(function, batched=True)

    return {
        "train": dataset_dict[dataset_info["train"]],
        "eval": dataset_dict[dataset_info["eval"]],
        "test": dataset_dict.get(dataset_info["test"]),
    }
