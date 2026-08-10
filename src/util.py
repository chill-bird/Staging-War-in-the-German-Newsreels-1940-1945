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

VIDEO_DATA_TSV = Path(os.getenv("VIDEO_DATA_TSV")).resolve()
CHAPTERS_DATA_TSV = Path(os.getenv("CHAPTERS_DATA_TSV")).resolve()
KEYFRAME_CLASSIFICATION_TSV = Path(os.getenv("KEYFRAME_CLASSIFICATION_TSV")).resolve()
DOC_DIR = Path(Path(__file__).parent.parent.resolve() / "doc").resolve()
NARRATIVES_DATA_TSV  = Path(os.getenv("NARRATIVES_DATA_TSV")).resolve()

assert VIDEO_DATA_TSV.is_file(), "Could not find video data TSV."
assert CHAPTERS_DATA_TSV.is_file(), "Could not find chapters data TSV."

CHAPTER_BLACKLIST = [
    {
        # Is map animation of combat developments
        "filestem": "591_1942",
        "chapter": "5",
    },
]

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
    len(df)

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

def get_cleaned_narratives_df() -> pd.DataFrame():
    # Load chapters
    tsv_file = Path(NARRATIVES_DATA_TSV).resolve()
    df = pd.read_csv(tsv_file, sep="\t")

    # Extract year and episode from filestem (e.g. "511_1940")
    df[["episode", "year"]] = df["filestem"].str.extract(r"(\d+)_(\d{4})").astype(int)

    # Sort chronologically
    df = df.sort_values("year")

    # Extract year and episode from filestem (e.g. "511_1940")
    df[["episode", "year"]] = df["filestem"].str.extract(r"(\d+)_(\d{4})").astype(int)

    # Sort chronologically
    df = df.sort_values("year")

    multi_label_cols = [
        "narrative_framing",
        "embodiment_mode",
        "legitimation_strategy",
        "enemy_moral_status",
        "actor_configuration",
        "violence_visibility"
    ]

    single_label_cols = [
        "agency_level"
    ]

    # clean faulty values
    df["violence_visibility"] = (
        df["violence_visibility"]
        .str.replace("i;m;p;l;i;e;d", "implied", regex=False)
        .str.replace("a;b;s;e;n;t", "absent", regex=False)
        .str.split(";")
        .apply(lambda x: ";".join(sorted(x)))
    )

    df["agency_level"] = (
        df["agency_level"]
        .str.replace("['medium']", "medium", regex=False)
        .str.replace("['high']", "high", regex=False)
        .str.split(";")
        .apply(lambda x: ";".join(sorted(x)))
    )

    for col in multi_label_cols:
        # Convert semicolon-separated strings to lists
        df[col] = df[col].str.split(";")
        
        # Remove whitespace and lowercase everything
        df[col] = df[col].apply(lambda lst: [x.strip().lower() for x in lst if x])
        
        # Sort the labels so that order does not matter
        df[col] = df[col].apply(lambda lst: sorted(lst))
        
        # Convert back to string for get_dummies
        df[col] = df[col].apply(lambda lst: ";".join(lst))

    for col in multi_label_cols:
        dummies = df[col].str.get_dummies(sep=";")
        dummies.columns = [f"{col}_{c}" for c in dummies.columns]
        df = pd.concat([df, dummies], axis=1)

    for col in single_label_cols:
        dummies = pd.get_dummies(df[col])
        dummies.columns = [f"{col}_{c}" for c in dummies.columns]
        df = pd.concat([df, dummies], axis=1)

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
