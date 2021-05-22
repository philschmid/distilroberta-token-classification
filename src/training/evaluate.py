import argparse
import logging

import torch
from datasets import load_metric
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.sampler import SequentialSampler
from transformers import AutoModelForTokenClassification, AutoTokenizer, DataCollatorForTokenClassification

from preprocess_utils import load_ner_dataset, tokenize_dataset
from train_utils import prepare_compute_metrics


# Set up logging
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser()
    # hyperparameters sent by the client are passed as command-line arguments to the script.
    parser.add_argument("--model_name_or_path", type=str)
    parser.add_argument("--dataset", type=str, default="conll2003")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--fp16", type=bool, default=True)

    args, _ = parser.parse_known_args()
    return args


def evaluate(args):
    # Load tokenizer and preprocess dataset

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=True, add_prefix_space=True)
    model = AutoModelForTokenClassification.from_pretrained(args.model_name_or_path)

    datasets, num_labels, label_to_id, label_list = load_ner_dataset(args.dataset)
    padding = "max_length" if args.pad_to_max_length else False
    dataset = tokenize_dataset(dataset=datasets["test"], tokenizer=tokenizer, padding=padding, label_to_id=label_to_id)

    data_collator = DataCollatorForTokenClassification(tokenizer, pad_to_multiple_of=8 if args.fp16 else None)

    sampler = SequentialSampler(dataset)

    dataloader = DataLoader(
        dataset,
        sampler=sampler,
        batch_size=args.batch_size,
        collate_fn=data_collator,
    )

    metric = load_metric("seqeval")
    compute_metrics = prepare_compute_metrics(metric, label_list)

    model.eval()
    for step, batch in enumerate(dataloader):
        with torch.no_grad():
            outputs = model(**batch)

        predictions = outputs.logits
        labels = batch["labels"]
        metric.add_batch(
            predictions=predictions,
            references=labels,
        )  # predictions and preferences are expected to be a nested list of labels, not label_ids

    result = compute_metrics()
    print(result)
    return result
