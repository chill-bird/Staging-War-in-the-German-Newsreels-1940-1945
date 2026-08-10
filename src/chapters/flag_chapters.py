"""
match_chapters_gpt.py
---

For each video, match audio transcriptions to chapter description by using OpenAI GPT
"""

import pandas as pd
from pathlib import Path
from dotenv import load_dotenv
import os
from openai import OpenAI

# Paths
load_dotenv()

CHAPTERS_DATA_TSV = Path(os.getenv("CHAPTERS_DATA_TSV")).resolve()
CHAPTERS_DIR = Path(os.getenv("CHAPTERS_DIR")).resolve()
PROMPT_FILE = Path(Path(__file__).parent.resolve() / "prompt_flag_chapters.txt").resolve()
assert CHAPTERS_DATA_TSV.is_file(), "Could not find TSV."
assert PROMPT_FILE.is_file(), "Could not find prompt file."
assert CHAPTERS_DIR.is_dir(), "Could not find CHAPTERS_DIR directory."

RESULT_DIR = Path(os.getenv("CHAPTERS_FLAGGED_DIR")).resolve()
RESULT_DIR.mkdir(parents=False, exist_ok=True)
assert RESULT_DIR.is_dir(), "Could not find directory for timestamped chapters."


with open(PROMPT_FILE, "r") as f:
    PROMPT_TEMPLATE = f.read()

# LLM
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
MODEL = "gpt-5-mini"
client = OpenAI()


def create_prompt(row: pd.Series) -> str:
    """Creates prompt for one episode"""

    chapter_name_placeholder = "$chapter$"
    chapter_description_placeholder = "$chapter_description$"
    audio_transcription_placeholder = "$audio_transcription$"
    filestem_placeholder = "$filestem$"
    chapter_name_str = row["chapter"]
    chapter_description_str = row["content"]
    audio_transcription_str = row["audio_transcription"]
    filestem_str = row["filestem"]

    prompt = PROMPT_TEMPLATE
    prompt = prompt.replace(chapter_name_placeholder, chapter_name_str)
    prompt = prompt.replace(audio_transcription_placeholder, audio_transcription_str)
    prompt = prompt.replace(chapter_description_placeholder, chapter_description_str)
    prompt = prompt.replace(filestem_placeholder, filestem_str)

    return prompt


def main() -> None:

    df = pd.read_csv(CHAPTERS_DATA_TSV, sep="\t")

    # Only use data where transcription timestamps exist
    df = df[
        (df["start"].notna())
        & (df["end"].notna())
        & (df["audio_transcription"].notna())
        & (df["content"].notna())
    ]

    for i, row in df.iterrows():
        print(f"[{i+1}/{len(df)}] Processing Chapter {row['chapter']} of {row['title']} ...")

        result_filename = row["filestem"] + "_" + str(row["chapter"]) + ".csv"
        result_filepath = Path(RESULT_DIR / result_filename).resolve()
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
