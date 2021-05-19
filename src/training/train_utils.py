import numpy as np

from transformers import AutoModelForTokenClassification


def prepare_compute_metrics(metric, label_list):
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

    return compute_metrics


def prepare_model_init(model_id, num_labels):
    def model_init():
        return AutoModelForTokenClassification.from_pretrained(
            model_id, return_dict=True, num_labels=num_labels, finetuning_task="ner"
        )

    return model_init
