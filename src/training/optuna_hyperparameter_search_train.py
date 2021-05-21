import argparse
import logging
import sys
import copy
import os
import json

import numpy as np
from datasets import load_metric

from preprocess_utils import load_ner_dataset, tokenize_dataset
from train_utils import prepare_compute_metrics, prepare_model_init


from transformers import (
    AutoTokenizer,
    DataCollatorForTokenClassification,
    Trainer,
    TrainingArguments,
    set_seed,
)
from transformers.trainer_utils import is_main_process


# Set up logging
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser()
    # hyperparameters sent by the client are passed as command-line arguments to the script.
    parser.add_argument("--n_trials", type=int, default=3)
    parser.add_argument("--per_device_train_batch_size", type=int, default=16)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=8)
    parser.add_argument("--model_name_or_path", type=str)
    parser.add_argument("--fp16", type=bool, default=True)
    parser.add_argument("--pad_to_max_length", type=bool, default=False)
    parser.add_argument("--output_dir", type=str, default="/opt/ml/model")

    # # Data, model, and output directories
    # parser.add_argument("--model_dir", type=str, default=os.environ["SM_MODEL_DIR"])
    # parser.add_argument("--training_dir", type=str, default=os.environ["SM_CHANNEL_TRAIN"])
    # parser.add_argument("--validation_dir", type=str, default=os.environ["SM_CHANNEL_VALIDATION"])
    # parser.add_argument("--test_dir", type=str, default=os.environ["SM_CHANNEL_TEST"])

    args, _ = parser.parse_known_args()
    return args


def main(args):
    # Evaluate during training and a bit more often
    # than the default to be able to prune bad trials early.
    # Disabling tqdm is a matter of preference.
    training_args = TrainingArguments(
        output_dir=args.output_dir,
    )
    # define training args
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        logging_dir=f"{args.output_dir}/logs",
        fp16=args.fp16,
        save_strategy="no",  # no checkpoint saving due to space
        # ray parameter
        learning_rate=1e-5,  # config
        num_train_epochs=2,  # config
        weight_decay=0.1,  # config
        evaluation_strategy="epoch",
        # evaluation_strategy="steps", eval_steps=500
        warmup_steps=0,
        do_train=True,
        do_eval=True,
        disable_tqdm=True,
    )

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    logger.setLevel(logging.INFO if is_main_process(training_args.local_rank) else logging.WARN)

    # Set seed before initializing model.
    set_seed(training_args.seed)

    # Load tokenizer and preprocess dataset

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=True, add_prefix_space=True)

    datasets, num_labels, label_to_id, label_list = load_ner_dataset("conll2003")
    padding = "max_length" if args.pad_to_max_length else False

    train_dataset = tokenize_dataset(
        dataset=datasets["train"], tokenizer=tokenizer, padding=padding, label_to_id=label_to_id
    )

    eval_dataset = tokenize_dataset(
        dataset=datasets["validation"], tokenizer=tokenizer, padding=padding, label_to_id=label_to_id
    )

    test_dataset = tokenize_dataset(
        dataset=datasets["test"], tokenizer=tokenizer, padding=padding, label_to_id=label_to_id
    )

    # Data collator
    data_collator = DataCollatorForTokenClassification(tokenizer, pad_to_multiple_of=8 if args.fp16 else None)

    # metric
    metric = load_metric("seqeval")
    objective_metric = "f1"
    compute_metrics = prepare_compute_metrics(metric, label_list)

    # load model
    model_init = prepare_model_init(args.model_name_or_path, num_labels)

    def my_hp_space(trial):
        # from ray import tune
        return {
            "learning_rate": trial.suggest_float("learning_rate", 1e-5, 5e-5, log=True),
            "num_train_epochs": trial.suggest_int("num_train_epochs", 3, 6),
            "weight_decay": trial.suggest_float("weight_decay", 0.0, 0.3),
        }

    def my_compute_objective(metrics):
        metrics = copy.deepcopy(metrics)
        return metrics[f"eval_{objective_metric}"]

    # Initialize our Trainer
    trainer = Trainer(
        args=training_args,
        tokenizer=tokenizer,
        model_init=model_init,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    best_run = trainer.hyperparameter_search(
        compute_objective=my_compute_objective,
        direction="maximize",
        hp_space=my_hp_space,
        backend="optuna",
        n_trials=args.n_trials,
    )
    logger.info(best_run)

    result = {
        objective_metric: best_run.objective,
        "hyperparameters": best_run.hyperparameters,
    }
    # Save predictions
    output_best_run = os.path.join(training_args.output_dir, "hpo_best_params.json")
    if trainer.is_world_process_zero():
        with open(output_best_run, "w") as outfile:
            json.dump(result, outfile)


if __name__ == "__main__":
    args = parse_args()

    main(args)
