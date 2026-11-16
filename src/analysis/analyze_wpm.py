"""Analyze chapter narration density with episode-clustered Gaussian GEE."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.analysis.fit_models import (
    CHAPTER_GROUPING,
    EPISODE_SCALE,
    QUADRATIC,
    analysis_report,
    analyze_outcome,
    grouped_figure_path,
)
from src.analysis.plots import (
    plot_wpm,
)
from src.util import DOC_DIR, get_filtered_chapters_df

FIG_WORD_COUNTS_FILE = Path(DOC_DIR / "imgs" / "word_counts_gee.pdf").resolve()


def prepare_wpm_data(chapters: pd.DataFrame) -> pd.DataFrame:
    data = chapters.copy()
    if data.duplicated(["filestem", "chapter"]).any():
        raise ValueError("Chapter identifiers must be unique.")
    data["chapter_uid"] = data["filestem"].astype(str) + "_" + data["chapter"].astype(str)
    duration = (
        pd.to_numeric(data["end"], errors="coerce") - pd.to_numeric(data["start"], errors="coerce")
    ) / 1000
    words = data["audio_transcription"].str.split().str.len()
    data["word_count_per_minute"] = words * 60 / duration.where(duration > 0)
    with np.errstate(divide="ignore", invalid="ignore"):
        data["log_wpm"] = np.log(data["word_count_per_minute"])
    return data


def main() -> None:
    with analysis_report("wpm", CHAPTER_GROUPING):
        print("Combat chapters; WPM uses chapter duration and transcription word count.")
        data = prepare_wpm_data(get_filtered_chapters_df("combat"))
        data, models, best = analyze_outcome(data, "log_wpm", "gaussian", CHAPTER_GROUPING)
        fig, _ = plot_wpm(
            df=data,
            quadratic_model=models[QUADRATIC],
            spline_model=None,
            knot=None,
            episode_scale=EPISODE_SCALE,
        )
        try:
            fig.savefig(grouped_figure_path(FIG_WORD_COUNTS_FILE, CHAPTER_GROUPING))
        finally:
            plt.close(fig)


if __name__ == "__main__":
    main()
