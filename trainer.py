from pathlib import Path, PurePath

from transformers import Trainer
from transformers.trainer_utils import get_last_checkpoint


class MyTrainer(Trainer):
    def my_train(self, model_name_or_path):
        resume_from_checkpoint = None
        if Path(self.args.output_dir).is_dir() and not self.args.overwrite_output_dir:
            resume_from_checkpoint = get_last_checkpoint(self.args.output_dir)
            if resume_from_checkpoint is not None:
                print(
                    f"Checkpoint detected, resuming training at {resume_from_checkpoint}.",
                    "To avoid this behavior, change the `--output_dir` or add `--overwrite_output_dir` to train from scratch.",
                )
            elif any(Path(self.args.output_dir).iterdir()):
                raise ValueError(
                    f"Output directory ({self.args.output_dir}) already exists and is not empty.",
                    "Use --overwrite_output_dir to overcome.",
                )
        if resume_from_checkpoint is None and Path(model_name_or_path).is_dir():
            resume_from_checkpoint = model_name_or_path

        metrics = self.train(resume_from_checkpoint=resume_from_checkpoint).metrics
        self._print("training", metrics)

        # Saves the tokenizer too for easy upload
        self.save_model()
        self.state.save_to_json(
            PurePath(self.args.output_dir).joinpath("trainer_state.json")
        )

    def my_evaluate(self, eval_dataset):
        eval_result = self.evaluate(eval_dataset=eval_dataset)
        self._print("evaluation", eval_result)

    def my_predict(self, test_dataset):
        metrics = self.predict(test_dataset=test_dataset).metrics
        self._print("prediction", metrics)

    def my_hyperparameter_search(self):
        hp_space = lambda trial: {
            "learning_rate": trial.suggest_float("learning_rate", 1e-6, 1e-4, log=True),
            "num_train_epochs": trial.suggest_int("num_train_epochs", 1, 5),
            "seed": trial.suggest_int("seed", 1, 40),
            "per_device_train_batch_size": trial.suggest_categorical(
                "per_device_train_batch_size", [4, 8, 16, 32]
            ),
        }

        best_run = self.hyperparameter_search(
            hp_space=hp_space, n_trials=10, direction="maximize"
        )

        for n, v in best_run.hyperparameters.items():
            setattr(self.args, n, v)

    def _print(self, name, dict):
        output_file = PurePath(self.args.output_dir).joinpath(f"{name}_results.txt")
        with open(output_file, "w") as file:
            print(f"***** {name.title()} results *****")
            for key, value in sorted(dict.items()):
                print(f"  {key} = {value}")
                print(f"{key} = {value}", file=file)
