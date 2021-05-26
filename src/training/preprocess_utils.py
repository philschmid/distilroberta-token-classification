import json

from datasets import ClassLabel, load_dataset, concatenate_datasets, load_from_disk
import datasets

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


feature_column = ["tokens", "ner_tags"]
split_list = ["train", "validation", "test"]
three_class_feature = datasets.Sequence(
    datasets.features.ClassLabel(
        names=[
            "O",
            "B-PER",
            "I-PER",
            "B-ORG",
            "I-ORG",
            "B-LOC",
            "I-LOC",
        ]
    )
)


def remove_columns_from_dataset_dict(dataset_dict, feature_columns):
    assert sorted(split_list) == sorted(
        list(dataset_dict.keys())
    ), "Dataset is not containing all splits for train,test,val"
    for split in split_list:
        remove_column_list = [col for col in list(dataset_dict[split].features) if col not in feature_column]
        dataset_dict[split] = dataset_dict[split].remove_columns(remove_column_list)
    return dataset_dict


def merging_all_splits_from_dataset_dict(dataset1, dataset2):
    for split in split_list:
        dataset1[split] = concatenate_datasets([dataset1[split], dataset2[split]])
    return dataset1


def change_label_to_zero(example):
    example["ner_tags"] = [0 if label == 7 or label == 8 else label for label in example["ner_tags"]]
    return example


def merge_datasets(conll, wikiann, class_num=4):
    # wikiann
    # downsizing the large test and validation set and merging it in to train
    additional_selected_validation_wikiann = wikiann["validation"].train_test_split(test_size=0.5)
    additional_selected_test_wikiann = wikiann["test"].train_test_split(test_size=0.5)
    wikiann["train"] = concatenate_datasets([additional_selected_test_wikiann["train"], wikiann["train"]])
    wikiann["train"] = concatenate_datasets([additional_selected_validation_wikiann["train"], wikiann["train"]])
    wikiann["validation"] = additional_selected_validation_wikiann["test"]
    wikiann["test"] = additional_selected_test_wikiann["test"]

    # removing columns for conll
    wikiann_cleaned = remove_columns_from_dataset_dict(wikiann, feature_column)
    # conll
    # removing columns
    conll_cleaned = remove_columns_from_dataset_dict(conll, feature_column)
    if class_num == 3:
        conll_cleaned["train"] = conll_cleaned["train"].map(change_label_to_zero, batched=True)
        conll_cleaned["test"] = conll_cleaned["test"].map(change_label_to_zero, batched=True)
        conll_cleaned["validation"] = conll_cleaned["validation"].map(change_label_to_zero, batched=True)

    wikiann_cleaned.save_to_disk("../data/wikiann")
    conll_cleaned.save_to_disk("../data/conll")
    # merging dataset
    loaded_conll = load_from_disk("../data/conll")
    loaded_wikiann = load_from_disk("../data/conll")

    merge_dataset = merging_all_splits_from_dataset_dict(loaded_conll, loaded_wikiann)
    if class_num == 3:
        for split in split_list:
            merge_dataset[split].features["ner_tags"] = three_class_feature
        return merge_dataset
    else:
        return merge_dataset


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


def load_ner_dataset(name, class_num=4):
    if name == "conll2003":
        datasets = load_dataset("conll2003")
    elif name == "conllpp":
        datasets = load_dataset("conllpp")
    elif name == "wikiann":
        datasets = load_dataset("wikiann", "en")
    elif name == "wikiann-conll2003":
        conll = load_dataset("conll2003")
        wikiann = load_dataset("wikiann", "en")
        datasets = merge_datasets(conll, wikiann, class_num)
    else:
        raise ValueError("Define either conll, wikiann or all as name")
    features = datasets["train"].features
    print(features)
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
