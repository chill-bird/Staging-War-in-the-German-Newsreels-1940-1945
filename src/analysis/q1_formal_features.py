"""
q1_formal_features.py
---

Analyze per-chapter formal feature averages of
- word count per minute
- shot scale proportion
"""

from pathlib import Path
import pandas as pd
from scipy.stats import pearsonr, spearmanr


from src.util import (
    KEYFRAME_CLASSIFICATION_TSV,
    DOC_DIR,
    get_filtered_chapters_df,
)
from src.analysis.q1_plots import (
    plot_word_counts_per_episode,
    plot_shot_duration_per_episode,
    plot_shot_scale_per_episode,
    plot_top_to_bottom_similarities,
)

# -----------------------------------------------------------------
# Constants
# -----------------------------------------------------------------


class ColumnStats:
    def __init__(self, df: pd.DataFrame, x_col: str, y_col: str, title: str | None):

        self.df = df[df[y_col].notna()].copy()

        x = self.df[x_col]
        y = self.df[y_col]

        self.pearson_r, self.pearson_p = pearsonr(x, y)
        self.spearman_r, self.spearman_p = spearmanr(x, y)
        self.mean = y.mean()
        self.median = y.median()
        self.std = y.std()
        self.n = len(self.df)

        if title:
            self.title = title
        else:
            self.title = ""

    def print_stats(self) -> None:
        print("-" * 10)
        print(f"{self.title}\nStatistics")
        print("-" * 10)
        print(f"n = {self.n}")
        print(f"Median: {self.median:.2f}")
        print(f"Mean: {self.mean:.2f}")
        print(f"Standard deviation: {self.std:.2f}")
        print(f"Pearson:\tr = {self.pearson_r}\tp={self.pearson_p}")
        print(f"Spearman:\tr = {self.spearman_r}\tp={self.spearman_p}")


FIG_WORD_COUNTS_FILE = Path(DOC_DIR / "imgs" / "word_counts.pdf").resolve()
FIG_SHOT_SCALE_FILE = Path(DOC_DIR / "imgs" / "shot_scale.pdf").resolve()
FIG_SHOT_DURATION_FILE = Path(DOC_DIR / "imgs" / "shot_duration.pdf").resolve()
FIG_TOP_TO_BOTTOM_CLASSES = Path(DOC_DIR / "imgs" / "top_to_bottom_sims_shot_scale.pdf").resolve()

DF = get_filtered_chapters_df("combat")
KEYFRAMES_DF = pd.read_csv(KEYFRAME_CLASSIFICATION_TSV, sep="\t")
KEYFRAMES_DF["frame"] = KEYFRAMES_DF["frame"].astype(int)

SIMILARITY_THRESHOLD = 0.18

# -----------------------------------------------------------------
# Helpers - Word Count
# -----------------------------------------------------------------


def count_words(s: str):
    """Counts words separated by whitespace in a string."""
    return len(s.split())


def get_duration_sec(row: pd.Series) -> float | None:
    """Computes duration in seconds between start and end timestamp in one chapter"""
    start_ms = row["start"]
    end_ms = row["end"]

    if pd.notna(start_ms) and pd.notna(end_ms):
        duration_sec = (float(end_ms) - float(start_ms)) / 1000
        return duration_sec
    else:
        return None


def words_per_minute(row: pd.Series) -> float | None:
    """Computes word count per minute in one chapter"""
    n_words = row["word_count"]
    duration_sec = row["duration_sec"]

    if pd.notna(duration_sec) and pd.notna(n_words):
        return n_words * 60 / duration_sec
    else:
        return None


# -----------------------------------------------------------------
# Helpers - Shot Duration
# -----------------------------------------------------------------


def avg_shot_duration_sec(row: pd.Series) -> float | None:
    """Computes average shot length per second"""

    n_shots = row["shot_count"]
    duration_sec = row["duration_sec"]

    if pd.notna(duration_sec) and pd.notna(n_shots):
        return duration_sec / n_shots
    else:
        return None


# -----------------------------------------------------------------
# Helpers - Shot Scale
# -----------------------------------------------------------------


def get_key_frames(row: pd.Series) -> pd.DataFrame | None:
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
    ].copy()

    # print(f"Found {len(keyframes_df)} matching keyframes")

    return keyframes_df


def get_long_shot_ratio(row: pd.Series) -> float | None:
    """Computes long shot ratio in one chapter"""

    keyframes_df = get_key_frames(row)
    if keyframes_df.empty:
        return None

    shots = len(keyframes_df[keyframes_df["shot_scale_class"] == "Long Shot"])
    return shots / len(keyframes_df)


def get_close_shot_ratio(row: pd.Series) -> float | None:
    """Computes close shot ratio in one chapter"""

    keyframes_df = get_key_frames(row)
    if keyframes_df.empty:
        return None

    shots = len(keyframes_df[keyframes_df["shot_scale_class"] == "Close-up"])
    return shots / len(keyframes_df)


# -----------------------------------------------------------------
# Main Analysis
# -----------------------------------------------------------------


def main():
    """Runs analysis on formal features and saves plots and prints statistics."""
    df = DF.copy()

    ### Word counts ###

    df["word_count"] = df["audio_transcription"].apply(count_words)
    df["duration_sec"] = df.apply(lambda row: get_duration_sec(row), axis=1)
    df["word_count_per_minute"] = df.apply(lambda row: words_per_minute(row), axis=1)

    word_count_stats_episode = ColumnStats(
        df, "episode", "word_count_per_minute", "Word count per Minute per Episode"
    )
    word_count_stats_episode.print_stats()

    plt1 = plot_word_counts_per_episode(
        df,
        word_count_stats_episode.spearman_p,
        word_count_stats_episode.spearman_r,
        word_count_stats_episode.median,
        word_count_stats_episode.mean,
        word_count_stats_episode.std,
    )
    plt1.savefig(FIG_WORD_COUNTS_FILE)

    ### Shot duration ###

    df["avg_shot_duration"] = df.apply(lambda row: avg_shot_duration_sec(row), axis=1)

    shot_duration_stats_episode = ColumnStats(
        df, "episode", "avg_shot_duration", "Average shot duration per Episode"
    )
    shot_duration_stats_episode.print_stats()

    plt2 = plot_shot_duration_per_episode(
        df,
        shot_duration_stats_episode.spearman_p,
        shot_duration_stats_episode.spearman_r,
        shot_duration_stats_episode.median,
        shot_duration_stats_episode.mean,
        shot_duration_stats_episode.std,
    )
    plt2.savefig(FIG_SHOT_DURATION_FILE)

    ### Shot scale ###

    df["ratio_close_shots"] = df.apply(lambda row: get_close_shot_ratio(row), axis=1)
    df["ratio_long_shots"] = df.apply(lambda row: get_long_shot_ratio(row), axis=1)

    df["ratio_diff"] = df.apply(
        lambda row: row["ratio_close_shots"] - row["ratio_long_shots"], axis=1
    )
    shot_scale_diff_stats_episode = ColumnStats(
        df, "episode", "ratio_diff", "Difference between long and close shots"
    )
    shot_scale_diff_stats_episode.print_stats()

    plt3 = plot_shot_scale_per_episode(
        df,
        shot_scale_diff_stats_episode.spearman_p,
        shot_scale_diff_stats_episode.spearman_r,
        shot_scale_diff_stats_episode.median,
        shot_scale_diff_stats_episode.mean,
        shot_scale_diff_stats_episode.std,
    )
    plt3.savefig(FIG_SHOT_SCALE_FILE)

    plt4 = plot_top_to_bottom_similarities(
        KEYFRAMES_DF[KEYFRAMES_DF["shot_scale_sim_score"] >= SIMILARITY_THRESHOLD]
    )
    plt4.savefig(FIG_TOP_TO_BOTTOM_CLASSES)


if __name__ == "__main__":
    main()
