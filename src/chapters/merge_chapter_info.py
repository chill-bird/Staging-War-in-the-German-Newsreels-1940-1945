"""
match_chapter_info.py
---

Merge information for chapters to one dataframe.
"""

import pandas as pd
from pathlib import Path
from dotenv import load_dotenv
import os
from src.util import (
    get_chapter_mapping_df,
    get_chapters_df,
    get_transcription_df,
    get_content_flags_df,
    get_scenes_df,
    mmss_to_ms,
    millisecs_to_frames,
)

# Paths
load_dotenv()

TSV = Path(os.getenv("VIDEO_DATA_TSV")).resolve()
CHAPTERS_DIR = Path(os.getenv("CHAPTERS_DIR")).resolve()
TRANSCRIPTIONS_DIR = Path(os.getenv("TRANSCRIPTIONS_DIR")).resolve()
CHAPTERS_TIMESTAMPED_DIR = Path(os.getenv("CHAPTERS_TIMESTAMPED_DIR")).resolve()
CHAPTERS_FLAGGED_DIR = Path(os.getenv("CHAPTERS_FLAGGED_DIR")).resolve()

assert TSV.is_file(), "Could not find TSV."
assert CHAPTERS_DIR.is_dir(), "Could not find chapters directory."
assert TRANSCRIPTIONS_DIR.is_dir(), "Could not find transcriptions directory."
assert CHAPTERS_TIMESTAMPED_DIR.is_dir(), "Could not find directory for timestamped chapters."
assert CHAPTERS_FLAGGED_DIR.is_dir(), "Could not find directory for flagged content chapters."

TARGET_FILE = Path(os.getenv("CHAPTERS_DATA_TSV")).resolve()

COLUMNS_TO_KEEP = [
    "title",
    "chapter",
    "start_mm:ss",
    "end_mm:ss",
    "is_war_report",
    "is_combat_scene",
    "german_soldiers_depicted",
    "shot_count",
    "content",
    "audio_transcription",
    "filestem",
    "episode",
    "year",
    "start",
    "end",
    "start_frame",
    "end_frame",
]


def get_transcription_for_chapter(
    df: pd.DataFrame, start: float | int | None, end: float | int | None
) -> str:
    if start is None or end is None or pd.isna(start) or pd.isna(end):
        return ""
    mask = (df["start"] <= end) & (df["end"] >= start)
    texts = (
        df.loc[mask, "text"]
        .dropna()  # removes NaN / pd.NA
        .astype(str)  # ensures strings
        .str.strip()  # removes ""
    )

    texts = texts[texts != ""]  # drop empty strings

    return " ".join(texts)


def get_shot_count_for_chapter(
    df: pd.DataFrame, start: float | int | None, end: float | int | None
) -> int | None:
    """Returns number of shots in one chapter."""
    if start is None or end is None or pd.isna(start) or pd.isna(end):
        return None
    return len(df[(df["start"] <= end) & (df["end"] >= start)])


def main() -> None:
    movies_df = pd.read_csv(TSV, sep="\t")

    # Only use data where transcription and chapters exist
    movies_df = movies_df[(movies_df["has_transcription"]) & (movies_df["has_chapters"])]

    chapters_dfs = []
    n = len(movies_df)

    for i, movie in movies_df.iterrows():
        print(f"[{i + 1}/{n}] {movie['title']}\n")

        # ------------------------------------------------------------------
        # Ground truth: chapters
        # ------------------------------------------------------------------
        chapters_df = get_chapters_df(movie["filestem"]).copy()
        chapters_df["chapter"] = chapters_df["chapter"].astype(str)

        chapters_df["title"] = movie["title"]
        chapters_df["filestem"] = movie["filestem"]
        chapters_df["episode"] = movie["episode"]
        chapters_df["year"] = movie["year"]

        # Enforce one row per chapter
        if not chapters_df["chapter"].is_unique:
            raise ValueError(f"Duplicate chapters in chapters_df for {movie['filestem']}")

        # ------------------------------------------------------------------
        # Timestamp & Frame mapping
        # ------------------------------------------------------------------
        timestamp_mapping_df = get_chapter_mapping_df(movie["filestem"])
        if timestamp_mapping_df is None or timestamp_mapping_df.empty:
            print("No timestamp mapping found.")
            continue

        # timestamp_mapping_df = timestamp_mapping_df.copy()
        timestamp_mapping_df["chapter"] = timestamp_mapping_df["chapter"].astype(str)

        # Enforce uniqueness
        if not timestamp_mapping_df["chapter"].is_unique:
            raise ValueError(f"Duplicate chapters in timestamp_mapping_df for {movie['filestem']}")

        chapters_df = chapters_df.merge(
            timestamp_mapping_df,
            on="chapter",
            how="left",
            validate="one_to_one",
        )

        chapters_df["start"] = chapters_df["start_mm:ss"].fillna("").apply(mmss_to_ms)
        chapters_df["end"] = chapters_df["end_mm:ss"].fillna("").apply(mmss_to_ms)

        chapters_df.loc[chapters_df["start"].notna(), "start_frame"] = chapters_df.loc[
            chapters_df["start"].notna(), "start"
        ].apply(lambda x: millisecs_to_frames(int(x), float(movie["fps"])))

        chapters_df.loc[chapters_df["end"].notna(), "end_frame"] = chapters_df.loc[
            chapters_df["end"].notna(), "end"
        ].apply(lambda x: millisecs_to_frames(int(x), float(movie["fps"])))

        # ------------------------------------------------------------------
        # Transcriptions
        # ------------------------------------------------------------------
        transcriptions_df = get_transcription_df(movie["filestem"])

        chapters_df["audio_transcription"] = chapters_df.apply(
            lambda r: get_transcription_for_chapter(transcriptions_df, r["start"], r["end"]),
            axis=1,
        )

        # ------------------------------------------------------------------
        # Content flags (aggregated to ONE row per chapter)
        # ------------------------------------------------------------------
        flags = []
        for chapter in chapters_df["chapter"]:
            df = pd.DataFrame(get_content_flags_df(movie["filestem"], chapter))
            if not df.empty:
                flags.append(df)

        if flags:
            content_flags_df = pd.concat(flags, ignore_index=True)
            content_flags_df["chapter"] = content_flags_df["chapter"].astype(str)

            content_flags_df = (
                content_flags_df[
                    [
                        "chapter",
                        "is_war_report",
                        "is_combat_scene",
                        "german_soldiers_depicted",
                    ]
                ]
                .groupby("chapter", as_index=False)
                .any()
            )

            chapters_df = chapters_df.merge(
                content_flags_df,
                on="chapter",
                how="left",
                validate="one_to_one",
            )
        else:
            chapters_df["is_war_report"] = False
            chapters_df["is_combat_scene"] = False
            chapters_df["german_soldiers_depicted"] = False

        # ------------------------------------------------------------------
        # Shot count
        # ------------------------------------------------------------------
        scenes_df = get_scenes_df(movie["filestem"])

        chapters_df["shot_count"] = chapters_df.apply(
            lambda r: get_shot_count_for_chapter(scenes_df, r["start"], r["end"]),
            axis=1,
        )

        chapters_dfs.append(chapters_df)

    # ----------------------------------------------------------------------
    # Final result
    # ----------------------------------------------------------------------
    result = pd.concat(chapters_dfs, ignore_index=True)
    result = result[COLUMNS_TO_KEEP]

    result = result.drop_duplicates()
    # Natural key validation
    if result.duplicated(subset=["filestem", "chapter"]).any():
        dupes = result[result.duplicated(subset=["filestem", "chapter"], keep=False)]
        raise ValueError(f"Duplicate (filestem, chapter) rows detected:\n{dupes}")

    result.to_csv(TARGET_FILE, index=False, sep="\t")


if __name__ == "__main__":
    main()
