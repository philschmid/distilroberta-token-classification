import argparse
import logging
import os
import sys

import numpy as np
import transformers
from datasets import load_metric
from transformers import AutoTokenizer, DataCollatorForTokenClassification, Trainer, TrainingArguments, set_seed
from transformers.trainer_utils import is_main_process

from preprocess_utils import change_entities, conll_id2label, conll_label2id, load_ner_dataset, tokenize_dataset
from train_utils import prepare_compute_metrics, prepare_model_init


# Set up logging
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser()
    # hyperparameters sent by the client are passed as command-line arguments to the script.
    parser.add_argument("--model_name_or_path", type=str)
    parser.add_argument("--dataset", type=str, default="conll2003")
    parser.add_argument("--task", type=str, default="ner")
    parser.add_argument("--num_train_epochs", type=float, default=3.0)
    parser.add_argument("--per_device_train_batch_size", type=int, default=8)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=8)
    parser.add_argument("--save_steps", type=int, default=2000)
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--fp16", type=bool, default=True)
    parser.add_argument("--pad_to_max_length", type=bool, default=False)
    parser.add_argument("--output_dir", type=str, default="/opt/ml/model")
    parser.add_argument("--use_auth_token", type=str, default="")
    parser.add_argument("--do_train", type=str, default="")
    parser.add_argument("--do_eval", type=str, default="")
    parser.add_argument("--do_test", type=str, default="")

    # # Data, model, and output directories
    # parser.add_argument("--model_dir", type=str, default=os.environ["SM_MODEL_DIR"])
    # parser.add_argument("--training_dir", type=str, default=os.environ["SM_CHANNEL_TRAIN"])
    # parser.add_argument("--validation_dir", type=str, default=os.environ["SM_CHANNEL_VALIDATION"])
    # parser.add_argument("--test_dir", type=str, default=os.environ["SM_CHANNEL_TEST"])

    args, _ = parser.parse_known_args()
    return args


def main(args):
    # define training args
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        weight_decay=args.weight_decay,
        logging_dir=f"{args.output_dir}/logs",
        learning_rate=float(args.learning_rate),
        save_steps=args.save_steps,
        fp16=args.fp16,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
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

    datasets, num_labels, label_to_id, label_list = load_ner_dataset(args.dataset)
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
    compute_metrics = prepare_compute_metrics(metric, label_list)

    # load model
    model_init = prepare_model_init(args.model_name_or_path, num_labels)
    model = model_init()

    # Initialize our Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    # Training
    if args.do_train != "":
        train_result = trainer.train()
        metrics = train_result.metrics
        trainer.save_model()  # Saves the tokenizer too for easy upload

        metrics["train_samples"] = len(train_dataset)
        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

    # Evaluation
    if args.do_eval != "":
        logger.info("*** Evaluate ***")

        metrics = trainer.evaluate()
        metrics["eval_samples"] = len(eval_dataset)
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    # Test
    if args.do_test != "":
        logger.info("*** Test ***")

        predictions, labels, metrics = trainer.predict(test_dataset, metric_key_prefix="test")
        predictions = np.argmax(predictions, axis=2)

        trainer.log_metrics("test", metrics)
        trainer.save_metrics("test", metrics)

    # # https://github.com/huggingface/transformers/blob/2582e59a57154ec5a71321eda24019dd94824e71/src/transformers/trainer.py#L2430
    if args.use_auth_token != "":
        # push to hub
        config_path = os.path.join(args.output_dir, "config.json")
        change_entities(config_path, conll_label2id, conll_id2label)

        kwargs = {
            "finetuned_from": args.model_name_or_path,
            "tags": "token-classification",
            "dataset_tags": args.dataset,
            "dataset": args.dataset,
        }

        trainer.push_to_hub(
            repo_name=f"{args.model_name_or_path}-{args.task}-{args.dataset}",
            use_auth_token=args.use_auth_token,
            **kwargs,
        )


if __name__ == "__main__":
    args = parse_args()

    main(args)
