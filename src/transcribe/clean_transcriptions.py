"""
clean_transcriptions.py
---

Transcribes audio files via whisper model
"""

from dotenv import load_dotenv
from pathlib import Path
import pandas as pd
import os

load_dotenv(dotenv_path=".env")  # Relative to root directory

TRANSCRIPTIONS_DIR = Path(os.getenv("TRANSCRIPTIONS_DIR")).resolve()
TRANSCRIPTIONS_CLEANED_DIR = Path(os.getenv("TRANSCRIPTIONS_CLEANED_DIR")).resolve()
assert TRANSCRIPTIONS_DIR.is_dir(), "Could not find transcriptions source directory."
assert (
    TRANSCRIPTIONS_DIR != TRANSCRIPTIONS_CLEANED_DIR
), "Transcription directory and cleaned transcription directory should be separate."

TSV_COLUMNS = ["start", "end", "text"]


def convert_ms_to_mmss(miliseconds: int) -> str:
    """Converts miliseconds to minute number format"""

    (seconds, miliseconds) = divmod(miliseconds, 1000)
    (minutes, seconds) = divmod(seconds, 60)
    return f"{minutes:02.0f}:{seconds:02.0f}"


def merge_comma_at_end(df: pd.DataFrame) -> pd.DataFrame:
    """Merge cells which end with comma with the next one
    Use start from the first and end from the second one)"""

    new_rows = []
    i = 0

    while i < len(df):
        current_row = df.iloc[i].copy()

        # Check if current row's text ends with comma and there's a next row
        if (
            i < len(df) - 1
            and isinstance(current_row["text"], str)
            and current_row["text"].rstrip().endswith(",")
        ):
            next_row = df.iloc[i + 1].copy()
            # Merge: remove comma from current, combine texts, use start from current and end from next
            merged_text = (
                current_row["text"].rstrip().rstrip(",").strip()
                + " "
                + str(next_row["text"]).strip()
            )
            merged_row = {
                "start": current_row["start"],
                "end": next_row["end"],
                "text": merged_text,
            }
            new_rows.append(merged_row)
            i += 2  # Skip both rows as we've merged them
        else:
            new_rows.append(current_row.to_dict())
            i += 1

    return pd.DataFrame(new_rows)


def clean_tsv_file(filepath: Path, target_dir: Path) -> None:
    """Applies data cleaning to one tsv file"""

    target_file = target_dir / filepath.name
    df = pd.read_csv(str(filepath), sep="\t")
    # Merge cells which end with comma with the next one
    df = merge_comma_at_end(df.copy())
    # Human readable timestamps
    df["start_mm:ss"] = df["start"].apply(convert_ms_to_mmss)
    df["end_mm:ss"] = df["end"].apply(convert_ms_to_mmss)
    df.to_csv(target_file, sep="\t", index=False)


def main():
    """Applies data cleaning to all tsv files in TRANSCRIPTIONS_DIR"""

    TRANSCRIPTIONS_CLEANED_DIR.mkdir(parents=False, exist_ok=True)

    tsv_files = sorted(list(TRANSCRIPTIONS_DIR.glob("*.tsv")))
    for tsv_file in tsv_files:
        print(tsv_file)
        clean_tsv_file(Path(tsv_file).resolve(), TRANSCRIPTIONS_CLEANED_DIR)


if __name__ == "__main__":
    main()
