import argparse
import logging
import sys

import numpy as np
from datasets import load_metric

import transformers
from ray import tune
from preprocess_utils import load_ner_dataset, tokenize_dataset
from train_utils import prepare_compute_metrics, prepare_model_init
from ray.tune.schedulers import PopulationBasedTraining


from transformers import (
    AutoTokenizer,
    DataCollatorForTokenClassification,
    Trainer,
    TrainingArguments,
    set_seed,
    AutoConfig,
    AutoModelForTokenClassification,
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
    parser.add_argument("--warmup_steps", type=int, default=500)
    parser.add_argument("--save_steps", type=int, default=500)
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
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        # ray parameter
        learning_rate=1e-5,  # config
        num_train_epochs=2,  # config
        weight_decay=0.1,  # config
        evaluation_strategy="epoch",
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

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    # Set the verbosity to info of the Transformers logger (on main process only):
    if is_main_process(training_args.local_rank):
        transformers.utils.logging.set_verbosity_info()
        transformers.utils.logging.enable_default_handler()
        transformers.utils.logging.enable_explicit_format()
    logger.info(f"Training/evaluation parameters {training_args}")

    # Set seed before initializing model.
    set_seed(training_args.seed)

    # Load tokenizer and preprocess dataset

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=True, add_prefix_space=True)

    datasets, num_labels, label_to_id, label_list = load_ner_dataset("conll")
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

    def compute_metrics(p):
        predictions, labels = p
        predictions = np.argmax(predictions, axis=2)

        # Remove ignored index (special tokens)
        true_predictions = [
            [label_list[p] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]
        true_labels = [
            [label_list[l] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]

        results = metric.compute(predictions=true_predictions, references=true_labels)
        return {
            "precision": results["overall_precision"],
            "recall": results["overall_recall"],
            "f1": results["overall_f1"],
            "accuracy": results["overall_accuracy"],
        }

    # load model
    model_init = prepare_model_init(args.model_name_or_path, num_labels)

    tune_config = {
        "num_train_epochs": tune.choice([3, 4, 5, 6]),
    }

    scheduler = PopulationBasedTraining(
        time_attr="training_iteration",
        metric="eval_f1",
        mode="max",
        perturbation_interval=1,
        hyperparam_mutations={
            "weight_decay": tune.uniform(0.0, 0.3),
            "learning_rate": tune.uniform(1e-5, 5e-5),
        },
    )

    reporter = CLIReporter(
        parameter_columns={"weight_decay": "w_decay", "learning_rate": "lr", "num_train_epochs": "num_epochs"},
        metric_columns=["eval_f1", "eval_precision", "eval_accuracy", "epoch", "training_iteration"],
    )

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

    best_trial = trainer.hyperparameter_search(
        hp_space=lambda _: tune_config,
        backend="ray",
        n_trials=num_samples,
        resources_per_trial={"cpu": 1, "gpu": 1},
        scheduler=scheduler,
        keep_checkpoints_num=1,
        checkpoint_score_attr="training_iteration",
        stop={"training_iteration": 1} if smoke_test else None,
        progress_reporter=reporter,
        local_dir="./ray",
        name="tune_transformer_pbt",
        log_to_file=True,
    )
    logger.info(best_trial)

    # Test
    logger.info("*** Test ***")

    predictions, labels, metrics = trainer.predict(test_dataset, metric_key_prefix="test")
    predictions = np.argmax(predictions, axis=2)

    trainer.log_metrics("test", metrics)
    trainer.save_metrics("test", metrics)


if __name__ == "__main__":
    args = parse_args()

    main(args)
