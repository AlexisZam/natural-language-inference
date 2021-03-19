from pathlib import Path

from transformers import Trainer
from transformers.trainer_utils import get_last_checkpoint


class MyTrainer(Trainer):
    def my_train(self, pretrained_model_name_or_path):  # FIXME
        resume_from_checkpoint = self._my_get_last_checkpoint(
            pretrained_model_name_or_path
        )

        metrics = self.train(resume_from_checkpoint=resume_from_checkpoint).metrics
        self._print_metrics("train", metrics)
        self.save_metrics("train", metrics)

        # Saves the tokenizer too for easy upload
        self.save_model()
        self.save_state()

    def my_evaluate(self):
        metrics = self.evaluate()
        self._print_metrics("eval", metrics)
        self.save_metrics("eval", metrics)

    def my_predict(self, test_dataset):
        metrics = self.predict(test_dataset).metrics
        self._print_metrics("test", metrics)
        self.save_metrics("test", metrics)

    def my_hyperparameter_search(self):
        hp_space = lambda trial: {
            "per_device_train_batch_size": trial.suggest_categorical(
                "per_device_train_batch_size", [4, 8, 16, 32]
            ),
            "learning_rate": trial.suggest_float("learning_rate", 1e-6, 1e-4, log=True),
            "num_train_epochs": trial.suggest_int("num_train_epochs", 1, 5),
            "seed": trial.suggest_int("seed", 1, 40),
        }

        best_run = self.hyperparameter_search(
            hp_space=hp_space, n_trials=self.args.n_trials, direction="maximize"
        )

        for n, v in best_run.hyperparameters.items():
            setattr(self.args, n, v)

    def _my_get_last_checkpoint(self, pretrained_model_name_or_path):
        # FIXME
        path = Path(self.args.output_dir)
        if path.is_dir() and not self.args.overwrite_output_dir:
            resume_from_checkpoint = get_last_checkpoint(self.args.output_dir)
            if resume_from_checkpoint is not None:
                print(
                    f"Checkpoint detected, resuming training at {resume_from_checkpoint}."
                )
                return resume_from_checkpoint
            if any(path.iterdir()):
                raise ValueError(
                    f"Output directory ({self.args.output_dir}) already exists and is not empty."
                )
        if Path(pretrained_model_name_or_path).is_dir():
            return pretrained_model_name_or_path

    @staticmethod
    def _print_metrics(split, metrics):
        print(f"***** {split.title()} results *****")
        for key, value in sorted(metrics.items()):
            print(f"  {key} = {value}")
