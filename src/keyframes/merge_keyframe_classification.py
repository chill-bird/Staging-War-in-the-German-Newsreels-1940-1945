"""
merge_keyframe_contents.py
---

Merge content scores for keyframes to one dataframe.
"""

import pandas as pd
from pathlib import Path
from dotenv import load_dotenv
import os

# Paths
load_dotenv()

CONTENT_CLASSIFICATIONS_DIR = Path(os.getenv("CONTENT_CLASSIFICATIONS_DIR")).resolve()
SHOT_SCALE_CLASSIFICATIONS_DIR = Path(os.getenv("SHOT_SCALE_CLASSIFICATIONS_DIR")).resolve()

assert (
    CONTENT_CLASSIFICATIONS_DIR.is_dir()
), "Could not find keyframe content classification directory."
assert (
    SHOT_SCALE_CLASSIFICATIONS_DIR.is_dir()
), "Could not find keyframe shot scale classification directory."

TARGET_FILE = Path(os.getenv("KEYFRAME_CLASSIFICATION_TSV")).resolve()

COLUMNS_TO_KEEP = [
    "filestem",
    "chapter",
    "frame",
    "content_class",
    "content_sim_score",
    "shot_scale_class",
    "shot_scale_sim_score"
]

def main() -> None:

    dfs_content = []
    for p in CONTENT_CLASSIFICATIONS_DIR.glob("*.csv"):
        df = pd.read_csv(p)
        dfs_content.append(df)

    df_content = pd.concat(dfs_content, ignore_index=True)
    df_content = df_content.rename(columns={"prediction": "content_class", "score": "content_sim_score"})

    dfs_shot_scale = []
    for p in SHOT_SCALE_CLASSIFICATIONS_DIR.glob("*.csv"):
        df = pd.read_csv(p)
        dfs_shot_scale.append(df)

    dfs_shot_scale = pd.concat(dfs_shot_scale, ignore_index=True)
    dfs_shot_scale = dfs_shot_scale.rename(columns={"prediction": "shot_scale_class", "score": "shot_scale_sim_score"})

    # Merge both frames at columns filestem, chapter, frame
    result = pd.merge(
        df_content,
        dfs_shot_scale,
        on=["filestem", "chapter", "frame"],
        how="outer", 
    )
    result = result[COLUMNS_TO_KEEP]

    result = result.drop_duplicates()
    result = result.sort_values(["filestem", "frame"])

    result.to_csv(TARGET_FILE, index=False, sep="\t")


if __name__ == "__main__":
    main()
