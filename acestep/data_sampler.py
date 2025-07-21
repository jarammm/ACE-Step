import json
from pathlib import Path


class DataSampler:
    def __init__(self, lora_dir="examples/pansori/input_params"):
        self.lora_input_params_files = list(Path(lora_dir).glob("*.json"))

    def load_json(self, file_path):
        with open(file_path, "r", encoding="utf-8") as f:
            return json.load(f)
    
    def sample_data(self, json_data):
        return (
            json_data["audio_duration"],
            json_data["prompt"],
            json_data["lyrics"],
            json_data["infer_step"],
            json_data["guidance_scale"],
            json_data["scheduler_type"],
            json_data["cfg_type"],
            json_data["omega_scale"],
            ", ".join(map(str, json_data["actual_seeds"])),
            json_data["guidance_interval"],
            json_data["guidance_interval_decay"],
            json_data["min_guidance_scale"],
            json_data["use_erg_tag"],
            json_data["use_erg_lyric"],
            json_data["use_erg_diffusion"],
            ", ".join(map(str, json_data["oss_steps"])),
            json_data["guidance_scale_text"] if "guidance_scale_text" in json_data else 0.0,
            (
                json_data["guidance_scale_lyric"]
                if "guidance_scale_lyric" in json_data
                else 0.0
            ),
            json_data["audio_path"],
            json_data["lora_name_or_path"],
            json_data["lora_weight"]
        )
    
    def sample(self):
        json_data_list = [self.load_json(json_path) for json_path in self.lora_input_params_files]
        sampled_json_data_list = [self.sample_data(json_data) for json_data in json_data_list]
        return sampled_json_data_list