import pandas as pd
from io import StringIO
from datasets import Dataset, DatasetDict

repo_path = "evilfreelancer/LOGIC-701-instruct"

LANGUAGE_CONFIG = {
    "en": {
        "input_prefix": "The following are possible answers, choose only one number of answer without additional reasoning and symbols:\n",
        "answer_prefix": "\nAnswer:",
    },
    "ru": {
        "input_prefix": "Ниже приведен список с вариантами ответов, выберите только одно число обозначающее ответ без дополнительных рассуждений и символов:\n",
        "answer_prefix": "\nОтвет:",
    },
}


def apply_language_mapping(example, config):
    example["input"] = config["input_prefix"] + example["input"] + config["answer_prefix"]
    return example


def push_dataset_to_hub(jsonl_file, repo_path: str, config_name: str, language_config: dict):
    df = pd.read_json(jsonl_file, lines=True)
    df.reset_index(drop=True, inplace=True)
    df.index.name = "id"
    dataset = Dataset.from_pandas(df)
    dataset = dataset.map(lambda x: apply_language_mapping(x, language_config))
    dataset_dict = DatasetDict({"train": dataset})
    dataset_dict.push_to_hub(repo_path, config_name=config_name)


if __name__ == "__main__":
    jsonl_file = open("converted_en_train.jsonl", "r")
    push_dataset_to_hub(jsonl_file=jsonl_file, repo_path=repo_path, config_name="en", language_config=LANGUAGE_CONFIG["en"])

    jsonl_file = open("converted_ru_train.jsonl", "r")
    push_dataset_to_hub(jsonl_file=jsonl_file, repo_path=repo_path, config_name="ru", language_config=LANGUAGE_CONFIG["ru"])
