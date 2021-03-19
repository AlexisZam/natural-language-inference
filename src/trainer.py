from transformers import Trainer
from transformers.trainer_utils import get_last_checkpoint


class MyTrainer(Trainer):
    def my_train(self):
        resume_from_checkpoint = (
            None
            if self.args.overwrite_output_dir
            else get_last_checkpoint(self.args.output_dir)
        )

        metrics = self.train(resume_from_checkpoint=resume_from_checkpoint).metrics
        self._print_metrics("train", metrics)
        self.save_metrics("train", metrics)

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

    @staticmethod
    def _print_metrics(split, metrics):
        print(f"***** {split.title()} results *****")
        for key, value in sorted(metrics.items()):
            print(f"  {key} = {value}")
