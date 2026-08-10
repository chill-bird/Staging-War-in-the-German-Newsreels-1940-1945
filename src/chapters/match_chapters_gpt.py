"""
match_chapters_gpt.py
---

For each video, match audio transcriptions to chapter description by using OpenAI GPT
"""

import pandas as pd
from pathlib import Path
from dotenv import load_dotenv
import os
from io import StringIO
from openai import OpenAI

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


with open(PROMPT_FILE, "r") as f:
    PROMPT_TEMPLATE = f.read()

# LLM
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
MODEL = "gpt-5-mini"
client = OpenAI()


def get_chapters_str(row: pd.Series) -> str:
    chapters_tsv = Path(CHAPTERS_DIR / (row["filestem"] + "_chapters.tsv")).resolve()
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

    return prompt


def main() -> None:
    df = pd.read_csv(TSV, sep="\t")
    # Only use data where transcription and chapters exist
    df = df[(df["has_transcription"]) & (df["has_chapters"])]
    # Existing chapter mappings


    for i, row in df.iterrows():
        print(f"[{i+1}/{len(df)}] Processing {row['title']} ...")

        result_filename = row["filestem"] + ".csv"
        result_filepath = Path(CHAPTERS_TIMESTAMPED_DIR / result_filename).resolve()
        if result_filepath.is_file():
            print("Skipping because file already exists.")
            continue

        prompt = create_prompt(row)
        try:
            response = client.responses.create(model=MODEL, input=prompt)
            result = response.output_text
            with open(result_filepath, "w", encoding="utf-8") as f:
                f.write(result)
            print(f'Processed {row["filestem"]}')
        except RuntimeError as e:
            print("ERROR: {e}")


if __name__ == "__main__":
    main()
