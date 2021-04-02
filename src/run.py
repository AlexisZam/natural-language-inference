#!/usr/bin/env python

from argparse import ArgumentParser
from datetime import datetime
from json import dump
from pathlib import PurePath
from shutil import rmtree

from datasets import DatasetDict, load_dataset, load_metric
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

    argument_group = argument_parser.add_mutually_exclusive_group(required=True)
    argument_group.add_argument("--do_hyperparameter_search", action="store_true")
    argument_group.add_argument("--do_train", action="store_true")

    argument_parser.add_argument("--do_predict", action="store_true")

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
    argument_group.add_argument("--n_trials", default=20, type=int)

    argument_group = argument_parser.add_argument_group(title="train")
    argument_group.add_argument("--overwrite_output_dir", action="store_true")
    argument_group.add_argument("--per_device_train_batch_size", default=8, type=int)
    argument_group.add_argument("--per_device_eval_batch_size", default=8, type=int)
    argument_group.add_argument("--learning_rate", default=5e-5, type=float)
    argument_group.add_argument("--save_total_limit", type=int)

    return argument_parser.parse_args()


def my_load_dataset(dataset_name, tokenizer, padding="do_not_pad", max_length=None):
    dataset_info = dataset_infos[dataset_name]

    function = lambda batch: tokenizer(
        batch[dataset_info.text],
        text_pair=batch[dataset_info.text_pair],
        padding=padding,
        truncation=True,
        max_length=max_length,
    )
    dataset_dict = (
        load_dataset(
            dataset_info.path, name=dataset_info.name, features=dataset_info.features
        )
        .filter(lambda example: example["label"] != -1)
        .map(function, batched=True)
    )

    return DatasetDict(
        {
            "train": dataset_dict[dataset_info.train],
            "validation": dataset_dict[dataset_info.validation],
            "test": dataset_dict.get(dataset_info.test),
        }
    )


class MyTrainer(Trainer):
    def my_hyperparameter_search(self, n_trials=20):
        hp_space = lambda trial: {
            "per_device_train_batch_size": trial.suggest_categorical(
                "per_device_train_batch_size", [4, 8, 16, 32]
            ),
            "learning_rate": trial.suggest_loguniform("learning_rate", 1e-6, 1e-4),
            "num_train_epochs": trial.suggest_int("num_train_epochs", 1, 5),
        }
        best_run = self.hyperparameter_search(
            hp_space=hp_space,
            compute_objective=lambda metrics: metrics["eval_accuracy"],
            n_trials=n_trials,
            direction="maximize",
        )
        file = PurePath(self.args.output_dir).joinpath("best_run.json")
        with open(file, mode="w") as fp:
            dump(best_run._asdict(), fp, indent=4)
        for run_id in range(n_trials):
            if run_id != int(best_run.run_id):
                rmtree(PurePath(self.args.output_dir).joinpath(f"run-{run_id}"))

    def my_predict(self, test_dataset):
        metrics = self.predict(test_dataset).metrics
        self.save_metrics("test", metrics)

    def my_train(self):
        metrics = self.train(
            resume_from_checkpoint=None if self.args.overwrite_output_dir else True,
        ).metrics
        self.save_metrics("train", metrics)

        self.save_state()

        metrics = self.evaluate()
        self.save_metrics("eval", metrics)


def main():
    set_seed(SEED)

    args = parse_args()

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
        overwrite_output_dir=args.overwrite_output_dir,
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

    dataset_dict = my_load_dataset(
        args.dataset_name, tokenizer, padding=args.padding, max_length=args.max_length
    )

    model_init = lambda: AutoModelForSequenceClassification.from_pretrained(
        args.pretrained_model_name_or_path,
        num_labels=dataset_dict["train"].features["label"].num_classes,
    )

    metric = load_metric("accuracy")
    compute_metrics = lambda eval_prediction: metric.compute(
        predictions=eval_prediction.predictions.argmax(axis=1),
        references=eval_prediction.label_ids,
    )

    trainer = MyTrainer(
        args=training_arguments,
        data_collator=data_collator,
        train_dataset=dataset_dict["train"],
        eval_dataset=dataset_dict["validation"],
        model_init=model_init,
        compute_metrics=compute_metrics,
    )

    if args.do_hyperparameter_search:
        trainer.my_hyperparameter_search(n_trials=args.n_trials)
    elif args.do_train:
        trainer.my_train()

    if args.do_predict:
        trainer.my_predict(dataset_dict["test"])


if __name__ == "__main__":
    main()
