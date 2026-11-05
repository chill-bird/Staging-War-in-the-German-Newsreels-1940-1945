"""
match_chapter_info.py
---

Merge information for chapters to one dataframe.
"""

import os
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

# Paths
load_dotenv()

DAT_DIR = Path(Path(__file__).parent.parent.resolve() / "dat").resolve()
VIDEO_DATA_TSV = Path(DAT_DIR / "video_data.tsv").resolve()
CHAPTERS_DATA_TSV = Path(DAT_DIR / "chapter_data.tsv").resolve()
KEYFRAME_CLASSIFICATION_TSV = Path(DAT_DIR / "keyframe_classification_data.tsv").resolve()
SCENES_DIR = Path(DAT_DIR / "scenes").resolve()
DOC_DIR = Path(Path(__file__).parent.parent.resolve() / "doc").resolve()
ANALYSIS_DIR = Path(Path(__file__).parent.parent.resolve() / "src" / "analysis").resolve()

assert VIDEO_DATA_TSV.is_file(), "Could not find video data TSV."
assert CHAPTERS_DATA_TSV.is_file(), "Could not find chapters data TSV."
assert KEYFRAME_CLASSIFICATION_TSV.is_file(), "Could not find keyframe data TSV."
assert SCENES_DIR.is_dir(), "Could not find scenes directory."

CHAPTER_BLACKLIST = [
    {
        # Is map animation of combat developments
        "filestem": "591_1942",
        "chapter": "5",
    },
]


def get_chapter_cuts(row: pd.Series) -> pd.DataFrame | None:
    """Returns shot boundaries within one chapter."""
    start = row.get("start_frame")
    end = row.get("end_frame")

    if start is None or end is None or pd.isna(start) or pd.isna(end):
        return None

    scenes_df = get_scenes_df(row["filestem"])
    if scenes_df is None:
        return None

    # Find scenes that overlap with the chapter boundaries
    mask = (scenes_df["start_frame"] <= end) & (scenes_df["end_frame"] >= start)

    return scenes_df[mask].copy()


def get_chapter_shot_durations_in_sec(row: pd.Series) -> list[float]:
    """Returns a list of shot durations or empty list for a chapter"""
    cuts_df = get_chapter_cuts(row)

    if cuts_df is None or cuts_df.empty:
        return []

    df_video = pd.read_csv(VIDEO_DATA_TSV, sep="\t")
    fps = df_video[df_video["filestem"] == row["filestem"]]["fps"].values[0]
    cuts_df["duration_ms"] = cuts_df.apply(
        lambda r: frames_to_millisecs(r["end_frame"] - r["start_frame"], fps) / 1000, axis=1
    )

    return cuts_df["duration_ms"].tolist()


def get_filtered_chapters_df(class_label: str) -> pd.DataFrame():
    """Returns chapter dataframe filtered for chapters classified with class_label."""
    assert class_label in [
        "combat",
        "war_report",
        "soldiers",
    ], 'filter keyword must be one of ["combat", "war_report", "soldiers"]'

    # Load chapters
    tsv_file = Path(CHAPTERS_DATA_TSV).resolve()
    df = pd.read_csv(tsv_file, sep="\t")

    # Apply blacklist
    for entry in CHAPTER_BLACKLIST:
        df = df[~((df["filestem"] == entry["filestem"]) & (df["chapter"] == entry["chapter"]))]

    # Only data with timestamps
    df = df[(df["start"].notna()) & (df["end"].notna())]

    match class_label:
        case "combat":
            df = df[(df["is_combat_scene"].notna()) & (df["is_combat_scene"])]
        case "war_report":
            df = df[(df["is_war_report"].notna()) & (df["is_war_report"])]
        case "soldiers":
            df = df[(df["german_soldiers_depicted"].notna()) & (df["german_soldiers_depicted"])]
        case _:
            raise AssertionError(
                'filter keyword must be one of ["combat", "war_report", "soldiers"]'
            )

    return df


def frames_to_timestamp(frame_no: int, fps: int | float) -> str:
    """Convert a frame number to mm:ss timestamp format in movie"""
    secs = round(frame_no / fps, 0)
    return seconds_to_mmss(secs)


def frames_to_millisecs(frame_no: int, fps: int | float) -> float:
    """Convert a frame number milliseconds"""
    return round((frame_no / fps) * 1000, 0)


def millisecs_to_frames(millisecs: int | float, fps: int) -> int:
    """Converts milliseconds to frame number."""
    return int(round(fps * (millisecs / 1000), 0))


def mmss_to_ms(mmss: str | None) -> int | None:
    """
    Convert a 'mm:ss' time string to total milliseconds.

    Example:
        '02:30' -> 150
    """
    if mmss in [None, ""] or (not isinstance(mmss, str) and mmss.isna()):
        return None
    try:
        minutes, seconds = mmss.split(":")
        return int(minutes) * 60 * 1000 + int(seconds) * 1000
    except (ValueError, AttributeError):
        raise ValueError(f"Invalid time format: {mmss!r}. Expected 'mm:ss'.")


def seconds_to_mmss(secs: int | float | None) -> str:
    """
    Convert total seconds to mm:ss format.

    Example:
        150 ->  '02:30'
    """
    if secs in [None, ""]:
        return ""
    try:
        total_seconds = int(secs)
        if total_seconds < 0:
            total_seconds = 0

        minutes, seconds = divmod(total_seconds, 60)
        return f"{minutes:02d}:{seconds:02d}"
    except (ValueError, AttributeError):
        raise ValueError(f"Invalid time format: {secs!r}. Expected 'mm:ss'.")


def get_chapter_mapping_df(filestem: str | None) -> pd.DataFrame | None:

    if not filestem:
        return None
    CHAPTERS_TIMESTAMPED_DIR = Path(os.getenv("CHAPTERS_TIMESTAMPED_DIR")).resolve()
    assert CHAPTERS_TIMESTAMPED_DIR.is_dir(), "Could not find directory for timestamped chapters."

    chapters_mapping_csv = Path(CHAPTERS_TIMESTAMPED_DIR / (filestem + ".csv")).resolve()
    if not chapters_mapping_csv.is_file():
        return None
    df_chapters = pd.read_csv(chapters_mapping_csv, sep=",")

    # Adjust timestamps based on chapter position
    n = len(df_chapters)
    last_idx = n - 1

    # Get video length from VIDEO_DATA_TSV
    df_video = pd.read_csv(VIDEO_DATA_TSV, sep="\t")
    video_row = df_video[df_video["filestem"] == filestem]
    if video_row.empty:
        raise ValueError(f"Could not find video data for filestem '{filestem}' in VIDEO_DATA_TSV.")
    video_length_sec = int(video_row["length"].iloc[0])

    for i in range(n):
        current_row = df_chapters.iloc[i]
        next_row = df_chapters.iloc[i + 1] if i < last_idx else None

        if i == last_idx:
            # Last chapter: subtract 4 seconds for credits screen
            end_secs = video_length_sec - 4
            df_chapters.at[current_row.name, "end_mm:ss"] = seconds_to_mmss(end_secs)
        elif next_row is not None and pd.notna(next_row.get("start_mm:ss", None)):
            # Has follow-up chapter: use next chapter's start timestamp
            df_chapters.at[current_row.name, "end_mm:ss"] = next_row["start_mm:ss"]
        else:
            # No follow-up chapter: clear both timestamps
            df_chapters.at[current_row.name, "start_mm:ss"] = ""
            df_chapters.at[current_row.name, "end_mm:ss"] = ""

    return df_chapters


def get_chapters_df(filestem: str | None) -> pd.DataFrame | None:

    if not filestem:
        return None
    CHAPTERS_DIR = Path(os.getenv("CHAPTERS_DIR")).resolve()
    assert CHAPTERS_DIR.is_dir(), "Could not find chapters directory."

    chapters_tsv = Path(CHAPTERS_DIR / (filestem + "_chapters.tsv")).resolve()
    if not chapters_tsv.is_file():
        return None
    df_chapters = pd.read_csv(chapters_tsv, sep="\t")
    return df_chapters


def get_transcription_df(filestem: str | None) -> pd.DataFrame | None:

    if not filestem:
        return None
    TRANSCRIPTIONS_DIR = Path(os.getenv("TRANSCRIPTIONS_DIR")).resolve()
    assert TRANSCRIPTIONS_DIR.is_dir(), "Could not find transcriptions directory."

    transcription_tsv = Path(TRANSCRIPTIONS_DIR / (filestem + ".tsv")).resolve()
    if not transcription_tsv.is_file():
        return None
    df_transcriptions = pd.read_csv(transcription_tsv, sep="\t")
    return df_transcriptions


def get_content_flags_df(filestem: str | None, chapter_name: str) -> pd.DataFrame | None:

    if not filestem or not chapter_name:
        return None
    CHAPTERS_FLAGGED_DIR = Path(os.getenv("CHAPTERS_FLAGGED_DIR")).resolve()
    assert CHAPTERS_FLAGGED_DIR.is_dir(), "Could not find directory for flagged content chapters."

    chapter_name = str(chapter_name)
    flags_csv = Path(CHAPTERS_FLAGGED_DIR / (filestem + "_" + chapter_name + ".csv")).resolve()
    if not flags_csv.is_file():
        return None
    df_flags = pd.read_csv(flags_csv, sep=",")
    return df_flags


def get_scenes_df(filestem: str | None) -> pd.DataFrame | None:

    if not filestem:
        return None
    SCENES_DIR = Path(os.getenv("SCENES_DIR")).resolve()
    assert SCENES_DIR.is_dir(), "Could not find directory for flagged content chapters."

    scenes_txt = Path(SCENES_DIR / (filestem + ".mp4.scenes.txt")).resolve()
    if not scenes_txt.is_file():
        return None

    df = pd.read_csv(scenes_txt, sep=" ", names=["start_frame", "end_frame"])
    df["start"] = df["start_frame"].apply(lambda x: frames_to_millisecs(x, 25))
    df["end"] = df["end_frame"].apply(lambda x: frames_to_millisecs(x, 25))
    # df["start_ts"] = df["start_frame"].apply(lambda x: frames_to_timestamp(x, 25))
    # df["end_ts"] = df["end_frame"].apply(lambda x: frames_to_timestamp(x, 25))
    return df


def get_keyframe_paths(filestem: str, start_frame: int, end_frame: int) -> list[Path]:
    """Returns list of paths to keyframes between start_frame and end_frame of movie belonging to filestem"""

    keyframes_dir = Path(os.getenv("KEYFRAMES_DIR")).resolve()
    movie_dir = keyframes_dir / filestem
    paths = []
    for p in movie_dir.glob("*.jpg"):
        try:
            frame_no = int(p.stem)
        except ValueError:
            continue  # skip non-numeric filenames

        if start_frame <= frame_no <= end_frame:
            paths.append(p)

    return sorted(paths)


def get_keyframe_path(filestem: str, frame: int) -> Path | None:
    """Returns path to keyframe image of frame number of movie belonging to filestem"""

    keyframes_dir = Path(os.getenv("KEYFRAMES_DIR")).resolve()
    movie_dir = keyframes_dir / filestem
    path = movie_dir / f"{frame}.jpg"
    if path.is_file():
        return path
    else:
        return None
