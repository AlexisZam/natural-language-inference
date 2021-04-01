#!/usr/bin/env python

from argparse import ArgumentParser
from datetime import datetime
from json import dump
from pathlib import PurePath
from shutil import rmtree

from datasets import ClassLabel, load_dataset, load_metric
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
    set_seed,
)

from dataset_info import dataset_infos

SEED = 42


def parse_args():
    argument_parser = ArgumentParser()

    argument_group = argument_parser.add_argument_group(title="model")
    argument_group.add_argument("--pretrained_model_name_or_path", required=True)

    argument_group = argument_parser.add_argument_group(title="dataset")
    argument_group.add_argument(
        "--dataset_name", choices=dataset_infos.keys(), required=True
    )

    argument_group = argument_parser.add_argument_group(title="tokenizer")
    argument_group.add_argument(
        "--padding", action="store_const", const="max_length", default="do_not_pad"
    )
    argument_group.add_argument("--max_length", type=int)

    argument_group = argument_parser.add_argument_group(title="hyperparameter_search")
    argument_group.add_argument("--do_hyperparameter_search", action="store_true")
    argument_group.add_argument("--n_trials", default=20, type=int)

    argument_group = argument_parser.add_argument_group(title="train")
    argument_group.add_argument("--do_train", action="store_true")
    argument_group.add_argument(
        "--resume_from_checkpoint", action="store_const", const=True
    )
    argument_group.add_argument("--per_device_train_batch_size", default=8, type=int)
    argument_group.add_argument("--per_device_eval_batch_size", default=8, type=int)
    argument_group.add_argument("--learning_rate", default=5e-5, type=float)
    argument_group.add_argument("--save_total_limit", type=int)

    argument_group = argument_parser.add_argument_group(title="predict")
    argument_group.add_argument("--do_predict", action="store_true")

    return argument_parser.parse_args()


def main():
    set_seed(SEED)

    args = parse_args()

    dataset_info = dataset_infos[args.dataset_name]

    output_dir = PurePath("tmp_trainer").joinpath(
        args.pretrained_model_name_or_path, args.dataset_name
    )
    logging_dir = PurePath("runs").joinpath(
        args.pretrained_model_name_or_path,
        args.dataset_name,
        datetime.now().isoformat(timespec="seconds"),
    )
    training_arguments = TrainingArguments(
        output_dir,
        # overwrite_output_dir=None if args.resume_from_checkpoint else True,
        do_train=args.do_train,
        do_eval=args.do_train,
        do_predict=args.do_predict,
        evaluation_strategy="epoch",
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        learning_rate=args.learning_rate,
        logging_dir=logging_dir,
        save_total_limit=args.save_total_limit,
        fp16=True,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        skip_memory_metrics=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(args.pretrained_model_name_or_path)

    data_collator = (
        None
        if args.padding == "max_length"
        else DataCollatorWithPadding(tokenizer, pad_to_multiple_of=8)
    )
    function = lambda batch: tokenizer(
        batch[dataset_info.text],
        text_pair=batch[dataset_info.text_pair],
        padding=args.padding,
        truncation=True,
        max_length=args.max_length,
    )
    dataset_dict = (
        load_dataset(dataset_info.path, name=dataset_info.name)
        .filter(lambda example: example["label"] != -1)
        .map(function, batched=True)
    )
    if args.dataset_name == "scitail":
        class_label = ClassLabel(names=("entails", "neutral"))

        def function(batch):
            batch["label"] = class_label.str2int(batch["label"])
            return batch

        dataset_dict = dataset_dict.map(function, batched=True)

    # train_dataset = dataset_dict[dataset_info.train].shard(num_shards, 1)

    num_labels = (
        2
        if args.dataset_name == "scitail"
        else dataset_dict[dataset_info.train].features["label"].num_classes
    )
    model_init = lambda: AutoModelForSequenceClassification.from_pretrained(
        args.pretrained_model_name_or_path, num_labels=num_labels
    )

    metric = load_metric("accuracy")
    compute_metrics = lambda eval_prediction: metric.compute(
        predictions=eval_prediction.predictions.argmax(axis=1),
        references=eval_prediction.label_ids,
    )

    trainer = Trainer(
        args=training_arguments,
        data_collator=data_collator,
        train_dataset=dataset_dict[dataset_info.train],
        eval_dataset=dataset_dict[dataset_info.validation],
        model_init=model_init,
        compute_metrics=compute_metrics,
    )

    if args.do_hyperparameter_search:
        hp_space = lambda trial: {
            "per_device_train_batch_size": trial.suggest_categorical(
                "per_device_train_batch_size", [4, 8, 16, 32]
            ),
            "learning_rate": trial.suggest_loguniform("learning_rate", 1e-6, 1e-4),
            "num_train_epochs": trial.suggest_int("num_train_epochs", 1, 5),
        }
        best_run = trainer.hyperparameter_search(
            hp_space=hp_space,
            compute_objective=lambda metrics: metrics["eval_accuracy"],
            n_trials=args.n_trials,
            direction="maximize",
        )
        file = PurePath(trainer.args.output_dir).joinpath("best_run.json")
        with open(file, mode="w") as fp:
            dump(best_run._asdict(), fp, indent=4)
        for run_id in range(args.n_trials):
            if run_id != int(best_run.run_id):
                rmtree(PurePath(trainer.args.output_dir).joinpath(f"run-{run_id}"))

    if args.do_train:
        metrics = trainer.train(
            resume_from_checkpoint=args.resume_from_checkpoint
        ).metrics
        trainer.save_metrics("train", metrics)

        trainer.save_state()

        metrics = trainer.evaluate()
        trainer.save_metrics("eval", metrics)

    if args.do_predict:
        metrics = trainer.predict(dataset_dict[dataset_info.test]).metrics
        trainer.save_metrics("test", metrics)


if __name__ == "__main__":
    main()
