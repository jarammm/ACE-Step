import argparse
import json
import random
import torchaudio
import os
from glob import glob

TEMPLATE_DIR = "examples/pansori"
SONG_DIR = "data/test_data"
OUTPUT_DIR = "outputs"


def load_lyrics(path):
    """Load lyrics text file."""
    with open(path, "r", encoding="utf-8") as f:
        return f.read().strip().replace("\r\n", "\n").replace("\r", "\n")


def load_prompt(path):
    """Load prompt text file, shuffle tags and return as comma-separated string."""
    with open(path, "r", encoding="utf-8") as f:
        tags = f.read().strip().split(",")
        random.shuffle(tags)
        return ",".join(tags)


def main():
    """
    This script generates JSON configuration files for all audio files in SONG_DIR.

    - For each .flac or .mp3 audio file in the directory, it:
        1. Extracts the base filename (title without extension)
        2. Loads matching _prompt.txt and _lyrics.txt
        3. Fills in a template JSON file with metadata
        4. Saves the new JSON file in examples/pansori/input_params/

    Arguments:
    --lora_name_or_path: LoRA model directory. (NOT exact path!)
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--lora_name_or_path",
        default="exps/sp_token/checkpoints/epoch=3-step=3575_lora",
        help="Path to the LoRA fine-tuned model to be used for generation."
    )
    args = parser.parse_args()

    template_path = os.path.join(TEMPLATE_DIR, "json_format.json")
    with open(template_path, "r", encoding="utf-8") as f:
        template_config = json.load(f)

    os.makedirs(os.path.join(TEMPLATE_DIR, "input_params"), exist_ok=True)

    # Gather all .flac and .mp3 files
    audio_files = glob(os.path.join(SONG_DIR, "*.flac")) + glob(os.path.join(SONG_DIR, "*.mp3"))

    for audio_file in audio_files:
        # Extract title by removing directory and extension
        title = os.path.splitext(os.path.basename(audio_file))[0]
        prompt_txt_path = os.path.join(SONG_DIR, f"{title}_prompt.txt")
        lyrics_txt_path = os.path.join(SONG_DIR, f"{title}_lyrics.txt")

        # Check if corresponding text files exist
        if not (os.path.exists(prompt_txt_path) and os.path.exists(lyrics_txt_path)):
            print(f"Skipping '{title}': prompt or lyrics file not found.")
            continue

        # Load audio
        audio, sr = torchaudio.load(audio_file)

        # Fill in the config
        config = template_config.copy()
        config["audio_duration"] = audio.shape[1] / sr
        config["lora_name_or_path"] = args.lora_name_or_path
        config["prompt"] = load_prompt(prompt_txt_path)
        config["lyrics"] = load_lyrics(lyrics_txt_path)
        config["wav_path"] = os.path.join(OUTPUT_DIR, f"{title}.wav")

        # Save JSON
        json_path = os.path.join(TEMPLATE_DIR, "input_params", f"{title}.json")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(config, f, ensure_ascii=False, indent=4)

        print(f"Config saved to {json_path}")


if __name__ == "__main__":
    main()
