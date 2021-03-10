from pathlib import Path, PurePath

from transformers import Trainer


class MyTrainer(Trainer):
    def my_train(self, last_checkpoint, model_name_or_path, output_dir):
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

        output_train_file = PurePath(output_dir).joinpath("train_results.txt")
        with open(output_train_file, "w") as writer:
            print("***** Train results *****")
            for key, value in sorted(metrics.items()):
                print(f"  {key} = {value}")
                writer.write(f"{key} = {value}\n")

        # Need to save the state, since Trainer.save_model saves only the tokenizer with the model
        self.state.save_to_json(PurePath(output_dir).joinpath("trainer_state.json"))

    def my_evaluate(self, task_name, eval_dataset, datasets, output_dir):
        print("*** Evaluate ***")

        # Loop to handle MNLI double evaluation (matched, mis-matched)
        tasks = [task_name]
        eval_datasets = [eval_dataset]
        if task_name == "mnli":
            tasks.append("mnli-mm")
            eval_datasets.append(datasets["validation_mismatched"])

        for eval_dataset, task in zip(eval_datasets, tasks):
            eval_result = self.evaluate(eval_dataset=eval_dataset)

            output_eval_file = PurePath(output_dir).joinpath(f"eval_results_{task}.txt")
            with open(output_eval_file, "w") as writer:
                print(f"***** Eval results {task} *****")
                for key, value in sorted(eval_result.items()):
                    print(f"  {key} = {value}")
                    writer.write(f"{key} = {value}\n")

    def my_predict(self, task_name, test_dataset, datasets, output_dir, label_list):
        print("*** Test ***")

        # Loop to handle MNLI double evaluation (matched, mis-matched)
        tasks = [task_name]
        test_datasets = [test_dataset]
        if task_name == "mnli":
            tasks.append("mnli-mm")
            test_datasets.append(datasets["test_mismatched"])

        for test_dataset, task in zip(test_datasets, tasks):
            # Removing the `label` columns because it contains -1 and Trainer won't like that.
            test_dataset.remove_columns_("label")
            predictions = self.predict(test_dataset=test_dataset).predictions
            predictions = predictions.argmax(axis=1)

            output_test_file = PurePath(output_dir).joinpath(f"test_results_{task}.txt")
            with open(output_test_file, "w") as writer:
                print(f"***** Test results {task} *****")
                writer.write("index\tprediction\n")
                for index, item in enumerate(predictions):
                    item = label_list[item]
                    writer.write(f"{index}\t{item}\n")
