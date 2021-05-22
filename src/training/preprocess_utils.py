import json

from datasets import ClassLabel, load_dataset


TEXT_COLUMN_NAME = "tokens"
LABEL_COLUMN_NAME = "ner_tags"

wikiann_id2label = {
    "0": "O",
    "1": "B-PER",
    "2": "I-PER",
    "3": "B-ORG",
    "4": "I-ORG",
    "5": "B-LOC",
    "6": "I-LOC",
}
wikiann_label2id = {
    "O": 0,
    "B-PER": 1,
    "I-PER": 2,
    "B-ORG": 3,
    "I-ORG": 4,
    "B-LOC": 5,
    "I-LOC": 6,
}

conll_id2label = {
    "0": "O",
    "1": "B-PER",
    "2": "I-PER",
    "3": "B-ORG",
    "4": "I-ORG",
    "5": "B-LOC",
    "6": "I-LOC",
    "7": "B-MISC",
    "8": "I-MISC",
}
conll_label2id = {
    "O": 0,
    "B-PER": 1,
    "I-PER": 2,
    "B-ORG": 3,
    "I-ORG": 4,
    "B-LOC": 5,
    "I-LOC": 6,
    "B-MISC": 7,
    "I-MISC": 8,
}


def change_entities(config_file, label2id, id2label):
    with open(config_file, "r") as file:
        json_data = json.load(file)
        print(json_data)
        json_data["label2id"] = label2id
        json_data["id2label"] = id2label
    with open(config_file, "w") as file:
        json.dump(json_data, file, indent=2)


def get_label_list(labels):
    unique_labels = set()
    for label in labels:
        unique_labels = unique_labels | set(label)
    label_list = list(unique_labels)
    label_list.sort()
    return label_list


def load_ner_dataset(name):
    if name == "conll2003":
        datasets = load_dataset("conll2003")
    elif name == "wikiann":
        datasets = load_dataset("wikiann", "en")
    elif name == "all":
        # TODO concat
        pass
    else:
        raise ValueError("Define either conll, wikiann or all as name")
    features = datasets["train"].features

    if isinstance(features[LABEL_COLUMN_NAME].feature, ClassLabel):
        label_list = features[LABEL_COLUMN_NAME].feature.names
        # No need to convert the labels since they are already ints.
        label_to_id = {i: i for i in range(len(label_list))}
    else:
        label_list = get_label_list(datasets["train"][LABEL_COLUMN_NAME])
        label_to_id = {l: i for i, l in enumerate(label_list)}
    num_labels = len(label_list)

    return datasets, num_labels, label_to_id, label_list


# Tokenize all texts and align the labels with them.
def tokenize_dataset(dataset, tokenizer, padding, label_to_id):
    def tokenize_and_align_labels(examples):
        tokenized_inputs = tokenizer(
            examples[TEXT_COLUMN_NAME],
            padding=padding,
            truncation=True,
            # We use this argument because the texts in our dataset are lists of words (with a label for each word).
            is_split_into_words=True,
        )
        labels = []
        for i, label in enumerate(examples[LABEL_COLUMN_NAME]):
            word_ids = tokenized_inputs.word_ids(batch_index=i)
            previous_word_idx = None
            label_ids = []
            for word_idx in word_ids:
                # Special tokens have a word id that is None. We set the label to -100 so they are automatically
                # ignored in the loss function.
                if word_idx is None:
                    label_ids.append(-100)
                # We set the label for the first token of each word.
                elif word_idx != previous_word_idx:
                    label_ids.append(label_to_id[label[word_idx]])
                # For the other tokens in a word, we set the label to either the current label or -100, depending on
                # the label_all_tokens flag.
                else:
                    label_ids.append(label_to_id[label[word_idx]])
                previous_word_idx = word_idx

            labels.append(label_ids)
        tokenized_inputs["labels"] = labels
        return tokenized_inputs

    return dataset.map(
        tokenize_and_align_labels,
        batched=True,
    )
