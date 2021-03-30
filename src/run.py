#!/usr/bin/env python

from argparse import ArgumentParser

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

    argument_parser.add_argument("--pretrained_model_name_or_path", required=True)

    argument_parser.add_argument(
        "--dataset_name", choices=dataset_infos.keys(), required=True
    )
    argument_parser.add_argument("--pad_to_max_length", action="store_true")
    argument_parser.add_argument("--max_length", type=int)

    argument_parser.add_argument("--do_hyperparameter_search", action="store_true")
    argument_parser.add_argument("--n_trials", default=20, type=int)

    argument_parser.add_argument("--do_train", action="store_true")
    argument_parser.add_argument("--resume_from_checkpoint", action="store_true")
    argument_parser.add_argument("--per_device_train_batch_size", default=8, type=int)
    argument_parser.add_argument("--per_device_eval_batch_size", default=8, type=int)
    argument_parser.add_argument("--learning_rate", default=5e-5, type=float)
    argument_parser.add_argument("--save_total_limit", type=int)

    argument_parser.add_argument("--do_predict", action="store_true")

    return argument_parser.parse_args()


def main():
    set_seed(SEED)

    args = parse_args()

    dataset_info = dataset_infos[args.dataset_name]

    training_arguments = TrainingArguments(
        f"/tmp/{args.pretrained_model_name_or_path}/{args.dataset_name}",
        evaluation_strategy="epoch",
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        learning_rate=args.learning_rate,
        save_total_limit=args.save_total_limit,
        fp16=True,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
    )

    tokenizer = AutoTokenizer.from_pretrained(args.pretrained_model_name_or_path)

    data_collator = (
        None
        if args.pad_to_max_length
        else DataCollatorWithPadding(tokenizer, pad_to_multiple_of=8)
    )

    function = lambda batch: tokenizer(
        batch[dataset_info.text],
        text_pair=batch[dataset_info.text_pair],
        padding="max_length" if args.pad_to_max_length else False,
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
        tokenizer=None if args.pad_to_max_length else tokenizer,
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
            hp_space=hp_space, n_trials=args.n_trials, direction="maximize"
        )
        print(best_run)

    if args.do_train:
        metrics = trainer.train(
            resume_from_checkpoint=True if args.resume_from_checkpoint else None
        ).metrics
        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)

        trainer.save_state()

        metrics = trainer.evaluate()
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    if args.do_predict:
        metrics = trainer.test(dataset_dict[dataset_info.test])
        trainer.log_metrics("test", metrics)
        trainer.save_metrics("test", metrics)


if __name__ == "__main__":
    main()
