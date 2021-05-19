import json


def change_entities(config_file, label2id, id2label):
    with open(config_file, "r") as file:
        json_data = json.load(file)
        print(json_data)
        json_data["label2id"] = label2id
        json_data["id2label"] = id2label
    with open(config_file, "w") as file:
        json.dump(json_data, file, indent=2)
