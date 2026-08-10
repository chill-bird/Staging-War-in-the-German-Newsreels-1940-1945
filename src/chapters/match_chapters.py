"""
match_chapters.py
---

For each video, match audio transcriptions to chapter description
"""

import pandas as pd
from pathlib import Path
from dotenv import load_dotenv
import os
from io import StringIO
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# Paths
load_dotenv()

TSV = Path(os.getenv("VIDEO_DATA_TSV")).resolve()
CHAPTERS_DIR = Path(os.getenv("CHAPTERS_DIR")).resolve()
TRANSCRIPTIONS_DIR = Path(os.getenv("TRANSCRIPTIONS_DIR")).resolve()
PROMPT_FILE = Path(Path(__file__).parent.resolve() / "prompt_match_chapters.txt").resolve()
CHAPTERS_TIMESTAMPED_DIR = Path(os.getenv("CHAPTERS_TIMESTAMPED_DIR")).resolve()
assert TSV.is_file(), "Could not find TSV."
assert PROMPT_FILE.is_file(), "Could not find prompt file."
assert CHAPTERS_DIR.is_dir(), "Could not find chapters directory."
assert TRANSCRIPTIONS_DIR.is_dir(), "Could not find transcriptions directory."
CHAPTERS_TIMESTAMPED_DIR.mkdir(parents=False, exist_ok=True)
assert CHAPTERS_TIMESTAMPED_DIR.is_dir(), "Could not find directory for timestamped chapters."
# Load prompt once
with open(PROMPT_FILE, "r") as f:
    PROMPT_TEMPLATE = f.read()


# LLM
MODEL_ID = "/work/ow52opul-wochenschau_analysis/models/qwen-3-14b"

print(f"Loading {MODEL_ID}")

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_ID,
    local_files_only=True,
)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    device_map="auto",
    local_files_only=True,
    torch_dtype="auto",
    # offload_folder=os.environ["MODEL_OFFLOAD_DIR"],  # Local folder
)

model.eval()
model.config.attn_implementation = "flash_attention_2"
# print(model.get_memory_footprint())


@torch.no_grad()
def run_inference(prompt: str) -> str:
    model_inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=False,
    ).to(model.device)

    generated_ids = model.generate(
        **model_inputs,
        max_new_tokens=1024,
        do_sample=False,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.eos_token_id,
    )

    output_ids = generated_ids[0][len(model_inputs.input_ids[0]) :].tolist()
    return tokenizer.decode(output_ids, skip_special_tokens=True).strip()


def get_chapters_str(row: pd.Series) -> str:
    chapters_tsv = Path(CHAPTERS_DIR / (row["filestem"] + "_chapters.tsv")).resolve()
    assert chapters_tsv.is_file()
    with open(chapters_tsv, "r") as f:
        chapters_str = f.read()
    return chapters_str


def get_transcription_str(row: pd.Series) -> str:
    transcription_tsv = Path(TRANSCRIPTIONS_DIR / (row["filestem"] + ".tsv")).resolve()
    assert transcription_tsv.is_file()
    df_transcriptions = pd.read_csv(transcription_tsv, sep="\t")
    df_transcriptions = df_transcriptions[["start_mm:ss", "end_mm:ss", "text"]]

    buffer = StringIO()
    df_transcriptions.to_csv(buffer, sep="\t", index=False)
    return buffer.getvalue()


def create_prompt(row: pd.Series) -> str:
    """Creates prompt for one episode"""

    chapters_placeholder = "$chapters$"
    chapters_str = get_chapters_str(row)
    audio_transcription_placeholder = "$audio_transcription$"
    audio_transcription_str = get_transcription_str(row)
    prompt = PROMPT_TEMPLATE
    prompt = prompt.replace(chapters_placeholder, chapters_str)
    prompt = prompt.replace(audio_transcription_placeholder, audio_transcription_str)

    print("### PROMPT ###")
    print(prompt)
    print("### PROMPT END ###")

    return prompt


def main() -> None:
    df = pd.read_csv(TSV, sep="\t")
    # Only use data where transcription and chapters exist
    df = df[(df["has_transcription"]) & (df["has_chapters"])]

    task_id = int(os.environ.get("SLURM_ARRAY_TASK_ID", 0))

    row = df.iloc[task_id]
    print(f"[{task_id+1}/{len(df)}] Processing {row['title']} ...")

    result_filename = row["filestem"] + ".csv"
    result_filepath = Path(CHAPTERS_TIMESTAMPED_DIR / result_filename).resolve()

    prompt = create_prompt(row)
    messages = [
        {
            "role": "system",
            "content": "You are a movie plot analyst. Output strict CSV only.",
        },
        {
            "role": "user",
            "content": prompt,
        },
    ]
    model_input = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,  # Switches between thinking and non-thinking modes. Default is True.
    )
    try:
        result = run_inference(model_input)
    except RuntimeError as e:
        result = f"ERROR: {e}"

    print(result)

    with open(result_filepath, "w", encoding="utf-8") as f:
        f.write(result)
    print(f'Processed {row["filestem"]}')


if __name__ == "__main__":
    main()
