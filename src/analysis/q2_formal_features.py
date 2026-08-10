"""
q2_formal_features.py
---

Analyze per-chapter formal feature averages of
- individual vs. collective soldier depictions
"""

from pathlib import Path
import pandas as pd


from src.util import (
    KEYFRAME_CLASSIFICATION_TSV,
    DOC_DIR,
    get_filtered_chapters_df,
)
from src.analysis.q1_formal_features import ColumnStats
from src.analysis.q2_plots import plot_depiction_ratio_per_episode, plot_top_to_bottom_similarities


# -----------------------------------------------------------------
# Constants
# -----------------------------------------------------------------

FIG_SOLDIER_DEPICTIONS_FILE = Path(DOC_DIR / "imgs" / "soldier_depictions.pdf").resolve()
FIG_TOP_TO_BOTTOM_CLASSES = Path(DOC_DIR / "imgs" / "top_to_bottom_sims_soldiers.pdf").resolve()

DF = get_filtered_chapters_df("soldiers")
KEYFRAMES_DF = pd.read_csv(KEYFRAME_CLASSIFICATION_TSV, sep="\t")
KEYFRAMES_DF["frame"] = KEYFRAMES_DF["frame"].astype(int)

SIMILARITY_THRESHOLD = 0.18

# -----------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------


def get_soldier_key_frames(row: pd.Series) -> pd.DataFrame | None:
    """Returns keyframes which match chapter"""

    filestem = row["filestem"]
    chapter = row["chapter"]
    start_frame = int(row["start_frame"])
    end_frame = int(row["end_frame"])
    keyframes_df = KEYFRAMES_DF[
        (KEYFRAMES_DF["filestem"] == filestem)
        & (KEYFRAMES_DF["chapter"] == chapter)
        & (KEYFRAMES_DF["frame"] >= start_frame)
        & (KEYFRAMES_DF["frame"] <= end_frame)
        & (KEYFRAMES_DF["shot_scale_sim_score"] >= SIMILARITY_THRESHOLD)
        & (
            (KEYFRAMES_DF["content_class"] == "individual")
            | (KEYFRAMES_DF["content_class"] == "multiples")
            | (KEYFRAMES_DF["content_class"] == "collective")
        )
    ].copy()

    return keyframes_df


def get_n_soldier_keyframes(keyframes_df: pd.DataFrame) -> int:
    """Returns total number of soldiers depicted"""
    n_soldier_keyframes = len(
        keyframes_df[
            (keyframes_df["content_class"] == "individual")
            | (keyframes_df["content_class"] == "multiples")
            | (keyframes_df["content_class"] == "collective")
        ]
    )
    return n_soldier_keyframes


def get_individuals_ratio(row: pd.Series) -> float | None:
    """Computes individual soldier depiction ratio of all soldier depictions in one chapter"""

    keyframes_df = get_soldier_key_frames(row)
    if keyframes_df.empty:
        return None
    n_soldier_keyframes = get_n_soldier_keyframes(keyframes_df)

    shots = len(keyframes_df[keyframes_df["content_class"] == "individual"])
    return shots / n_soldier_keyframes


def get_groups_ratio(row: pd.Series) -> float | None:
    """Computes group soldier depiction ratio of all soldier depictions in one chapter"""

    keyframes_df = get_soldier_key_frames(row)
    if keyframes_df.empty:
        return None
    n_soldier_keyframes = get_n_soldier_keyframes(keyframes_df)

    shots = len(keyframes_df[keyframes_df["content_class"] == "multiples"])
    return shots / n_soldier_keyframes


def get_collectives_ratio(row: pd.Series) -> float | None:
    """Computes collective soldier depiction ratio of all soldier depictions in one chapter"""

    keyframes_df = get_soldier_key_frames(row)
    if keyframes_df.empty:
        return None
    n_soldier_keyframes = get_n_soldier_keyframes(keyframes_df)

    shots = len(keyframes_df[keyframes_df["content_class"] == "collective"])
    return shots / n_soldier_keyframes


# -----------------------------------------------------------------
# Main Analysis
# -----------------------------------------------------------------


def main():
    """Runs analysis on formal features and saves plots and prints statistics."""

    df = DF.copy()

    df["ratio_individual_depictions"] = df.apply(lambda row: get_individuals_ratio(row), axis=1)
    df["ratio_group_depictions"] = df.apply(lambda row: get_groups_ratio(row), axis=1)
    df["ratio_collective_depictions"] = df.apply(lambda row: get_collectives_ratio(row), axis=1)

    df["ratio_diff"] = df.apply(
        lambda row: row["ratio_individual_depictions"] - row["ratio_collective_depictions"], axis=1
    )
    depictions_diff_stats_episode = ColumnStats(
        df, "episode", "ratio_diff", "Difference between individual and collective soldier depictions"
    )
    depictions_diff_stats_episode.print_stats()

    plt1 = plot_depiction_ratio_per_episode(
        df,
        depictions_diff_stats_episode.spearman_p,
        depictions_diff_stats_episode.spearman_r,
        depictions_diff_stats_episode.median,
        depictions_diff_stats_episode.mean,
        depictions_diff_stats_episode.std,
    )
    plt1.savefig(FIG_SOLDIER_DEPICTIONS_FILE)

    plt2 = plot_top_to_bottom_similarities(
        KEYFRAMES_DF[KEYFRAMES_DF["content_sim_score"] >= SIMILARITY_THRESHOLD]
    )
    plt2.savefig(FIG_TOP_TO_BOTTOM_CLASSES)


if __name__ == "__main__":
    main()
