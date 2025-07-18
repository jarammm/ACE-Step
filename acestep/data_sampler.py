import json
from pathlib import Path
import random


DEFAULT_ROOT_DIR = "examples/default/input_params"

class DataSampler:
    def __init__(self, root_dir=DEFAULT_ROOT_DIR, lora_dir=None):
        if lora_dir and lora_dir != "none":
            self.lora_input_params_files = list(Path(lora_dir).glob("*.json"))
        else:
            self.root_dir = root_dir
            self.input_params_files = list(Path(self.root_dir).glob("*.json"))

    def load_json(self, file_path):
        with open(file_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def sample(self):
        if self.lora_input_params_files:
            json_path = random.choice(self.lora_input_params_files)
            json_data = self.load_json(json_path)
            # Update the lora_name in the json_data
            json_data["lora_name_or_path"] = self.lora_input_params_files
        else:
            json_path = random.choice(self.input_params_files)
            json_data = self.load_json(json_path)
        
        return json_data
