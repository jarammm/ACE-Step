from tokenizers import Tokenizer
import os
from pathlib import Path
import yaml


DEFAULT_VOCAB_NAME = "vocab"

def load_yaml(vocab_yaml):
    with open(vocab_yaml, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
        return config

def get_vocab_file_path(vocab_name="vocab"):
    return os.path.join(
    os.path.dirname(os.path.realpath(__file__)), f"vocab_json/{vocab_name}.json"
)

def get_vocab_yaml_path(yaml_name):
    return os.path.join(
    os.path.dirname(os.path.realpath(__file__)), f"vocab_yaml/{yaml_name}.yaml"
)

def create_vocab_json(vocab):
    vocab_yaml = get_vocab_yaml_path(vocab)
    assert Path(get_vocab_yaml_path(vocab)).exists(), f"This path doesn't exist!"
    assert Path(f"{vocab}.json").exists() is True, f"Vocab json with this setting already exists!"
    config = load_yaml(vocab_yaml)
    
    vocab_file = get_vocab_file_path()
    tokenizer = Tokenizer.from_file(vocab_file)
    keys = list(config.keys())
    for key in keys:
        if key=="special_tokens":
            tokenizer.add_special_tokens(config["special_tokens"])
        elif key=="new_tokens":
            tokenizer.add_tokens(config["new_tokens"])
    
    json_path = get_vocab_file_path(vocab)
    tokenizer.save(json_path)
    print(f"new vocab json for voice tokenizer create in: {json_path}!")


if __name__=="__main__":
    test_name = "pansori_vocab"
    create_vocab_json(test_name)