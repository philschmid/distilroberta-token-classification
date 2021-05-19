import logging
import os
import argparse
import sys
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from datasets import ClassLabel, load_dataset, load_metric

import transformers
from transformers import (
    AutoConfig,
    AutoModelForTokenClassification,
    AutoTokenizer,
    DataCollatorForTokenClassification,
    HfArgumentParser,
    PreTrainedTokenizerFast,
    Trainer,
    TrainingArguments,
    set_seed,
)
from transformers.trainer_utils import get_last_checkpoint, is_main_process
from preprocess_utils import load_ner_dataset, tokenize_dataset
from train_utils import prepare_compute_metrics, prepare_model_init
import torch

# Set up logging
logger = logging.getLogger(__name__)

logging.basicConfig(
    level=logging.getLevelName("INFO"),
    handlers=[logging.StreamHandler(sys.stdout)],
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)


def parse_args():
    parser = argparse.ArgumentParser()
    # hyperparameters sent by the client are passed as command-line arguments to the script.
    parser.add_argument("--num_train_epochs", type=int, default=3)
    parser.add_argument("--per_device_train_batch_size", type=int, default=8)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=8)
    parser.add_argument("--warmup_steps", type=int, default=500)
    parser.add_argument("--save_steps", type=int, default=500)
    parser.add_argument("--model_name_or_path", type=str)
    parser.add_argument("--learning_rate", type=str, default=5e-5)
    parser.add_argument("--fp16", type=bool, default=True)
    parser.add_argument("--output_dir", type=str, default="/opt/ml/model")

    # # Data, model, and output directories
    # parser.add_argument("--model_dir", type=str, default=os.environ["SM_MODEL_DIR"])
    # parser.add_argument("--training_dir", type=str, default=os.environ["SM_CHANNEL_TRAIN"])
    # parser.add_argument("--validation_dir", type=str, default=os.environ["SM_CHANNEL_VALIDATION"])
    # parser.add_argument("--test_dir", type=str, default=os.environ["SM_CHANNEL_TEST"])

    args, _ = parser.parse_known_args()
    return args


def main(args):
    # Load tokenizer and preprocess dataset

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=True, add_prefix_space=True)

    datasets, num_labels, label_to_id, label_list = load_ner_dataset("conll")
    padding = True

    train_dataset = tokenize_dataset(
        dataset=datasets["train"], tokenzier=tokenizer, padding=padding, label_to_id=label_to_id
    )

    eval_dataset = tokenize_dataset(
        dataset=datasets["validation"], tokenzier=tokenizer, padding=padding, label_to_id=label_to_id
    )

    test_dataset = tokenize_dataset(
        dataset=datasets["test"], tokenzier=tokenizer, padding=padding, label_to_id=label_to_id
    )

    # Data collator
    data_collator = DataCollatorForTokenClassification(tokenizer, pad_to_multiple_of=8 if args.fp16 else None)

    # metric
    metric = load_metric("seqeval")
    compute_metrics = prepare_compute_metrics(metric)

    # load model
    model_init = prepare_model_init(args.model_name_or_path, num_labels)
    model = model_init()

    # define training args
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        warmup_steps=args.warmup_steps,
        evaluation_strategy="epoch",
        logging_dir=f"{args.output_dir}/logs",
        learning_rate=float(args.learning_rate),
        save_steps=2500,
        fp16=args.fp16,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
    )

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
    train_result = trainer.train()
    metrics = train_result.metrics
    trainer.save_model()  # Saves the tokenizer too for easy upload

    metrics["train_samples"] = len(train_dataset)
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()

    # Evaluation
    logger.info("*** Evaluate ***")

    metrics = trainer.evaluate()
    metrics["eval_samples"] = len(eval_dataset)
    trainer.log_metrics("eval", metrics)
    trainer.save_metrics("eval", metrics)

    # Test
    logger.info("*** Test ***")

    predictions, labels, metrics = trainer.predict(test_dataset, metric_key_prefix="test")
    predictions = np.argmax(predictions, axis=2)

    # Remove ignored index (special tokens)
    true_predictions = [
        [label_list[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]

    trainer.log_metrics("test", metrics)
    trainer.save_metrics("test", metrics)


if __name__ == "__main__":
    args = parse_args()

    main(args)
