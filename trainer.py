from pathlib import Path, PurePath

from transformers import Trainer
from transformers.trainer_utils import get_last_checkpoint


class MyTrainer(Trainer):
    def my_train(self, model_name_or_path):
        # Detecting last checkpoint.
        last_checkpoint = None
        if Path(self.args.output_dir).is_dir() and not self.args.overwrite_output_dir:
            last_checkpoint = get_last_checkpoint(self.args.output_dir)
            if last_checkpoint is None and any(Path(self.args.output_dir).iterdir()):
                raise ValueError(
                    f"Output directory ({self.args.output_dir}) already exists and is not empty.",
                    "Use --overwrite_output_dir to overcome.",
                )
            elif last_checkpoint is not None:
                print(
                    f"Checkpoint detected, resuming training at {last_checkpoint}.",
                    "To avoid this behavior, change the `--output_dir` or add `--overwrite_output_dir` to train from scratch.",
                )

        resume_from_checkpoint = (
            last_checkpoint
            if last_checkpoint is not None
            else model_name_or_path
            if Path(model_name_or_path).is_dir()
            else None
        )
        train_result = self.train(resume_from_checkpoint=resume_from_checkpoint)
        metrics = train_result.metrics

        self.save_model()  # Saves the tokenizer too for easy upload

        output_train_file = PurePath(self.args.output_dir).joinpath(
            "training_results.txt"
        )
        with open(output_train_file, "w") as writer:
            print("***** Training results *****")
            for key, value in sorted(metrics.items()):
                print(f"  {key} = {value}")
                writer.write(f"{key} = {value}\n")

        # Need to save the state, since Trainer.save_model saves only the tokenizer with the model
        self.state.save_to_json(
            PurePath(self.args.output_dir).joinpath("trainer_state.json")
        )

    def my_evaluate(self, task_name, eval_dataset, datasets):
        tasks = [task_name]
        eval_datasets = [eval_dataset]
        if task_name == "mnli":
            tasks.append("mnli-mm")
            eval_datasets.append(datasets["validation_mismatched"])

        for eval_dataset, task in zip(eval_datasets, tasks):
            eval_result = self.evaluate(eval_dataset=eval_dataset)

            output_eval_file = PurePath(self.args.output_dir).joinpath(
                f"evaluation_results_{task}.txt"
            )
            with open(output_eval_file, "w") as writer:
                print(f"***** Evaluation results {task} *****")
                for key, value in sorted(eval_result.items()):
                    print(f"  {key} = {value}")
                    writer.write(f"{key} = {value}\n")

    def my_predict(self, task_name, test_dataset, datasets, label_list):
        tasks = [task_name]
        test_datasets = [test_dataset]
        if task_name == "mnli":
            tasks.append("mnli-mm")
            test_datasets.append(datasets["test_mismatched"])

        for test_dataset, task in zip(test_datasets, tasks):
            # Removing the `label` columns because it contains -1 and Trainer won't like that.
            test_dataset = test_dataset.remove_columns("label")
            predictions = self.predict(test_dataset=test_dataset).predictions
            predictions = predictions.argmax(axis=1)

            output_test_file = PurePath(self.args.output_dir).joinpath(
                f"prediction_results_{task}.txt"
            )
            with open(output_test_file, "w") as writer:
                print(f"***** Prediction results {task} *****")
                writer.write("index\tprediction\n")
                for index, item in enumerate(predictions):
                    writer.write(f"{index}\t{label_list[item]}\n")
