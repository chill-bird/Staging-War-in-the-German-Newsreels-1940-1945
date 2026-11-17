"""Analyze chapter-average or nested individual shot lengths with Gaussian GEE."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.analysis.fit_models import (
    CHAPTER_GROUPING,
    AUTOMATED_KNOT,
    EPISODE_SCALE,
    LINEAR,
    NESTED_GROUPING,
    analysis_report,
    analyze_outcome,
    grouped_figure_path,
    validate_grouping,
)
from src.analysis.plots import (
    plot_asl,
)
from src.util import (
    DOC_DIR,
    SCENES_DIR,
    get_chapter_shot_durations_in_sec,
    get_filtered_chapters_df,
)

FIG_SHOT_DURATION_FILE = Path(DOC_DIR / "imgs" / "shot_duration_gee.pdf").resolve()


def prepare_asl_data(
    chapters: pd.DataFrame, grouping_level: str = CHAPTER_GROUPING
) -> pd.DataFrame:
    validate_grouping(grouping_level)
    data = chapters.copy()
    if data.duplicated(["filestem", "chapter"]).any():
        raise ValueError("Chapter identifiers must be unique.")
    data["chapter_uid"] = data["filestem"].astype(str) + "_" + data["chapter"].astype(str)
    durations = data.apply(
        lambda chapter: get_chapter_shot_durations_in_sec(chapter, scenes_dir=SCENES_DIR), axis=1
    )

    def positive_durations(values):
        values = np.asarray(values, dtype=float)
        return values[np.isfinite(values) & (values > 0)]

    durations = durations.apply(positive_durations)
    if grouping_level == NESTED_GROUPING:
        data["durations"] = durations
        missing = int(durations.apply(len).eq(0).sum())
        print(f"Excluded {missing} chapters without positive finite shot durations.")
        data = data.loc[durations.apply(len).gt(0)].explode("durations", ignore_index=True)
        data["durations"] = pd.to_numeric(data["durations"])
        data["log_shot_length"] = np.log(data["durations"])
        return data

    data["average_shot_length"] = durations.apply(
        lambda values: float(values.mean()) if values.size else np.nan
    )
    data["log_asl"] = np.log(data["average_shot_length"])
    return data


def main(grouping_level: str = CHAPTER_GROUPING) -> None:
    with analysis_report("asl", grouping_level):
        if grouping_level == NESTED_GROUPING:
            print("Combat chapters; each positive shot duration is logged before modeling.")
        else:
            print("Combat chapters; ASL is the mean of positive shot durations before taking the log.")
        print("Shots overlapping chapter boundaries are included.")
        data = prepare_asl_data(get_filtered_chapters_df("combat"), grouping_level)
        outcome = "log_shot_length" if grouping_level == NESTED_GROUPING else "log_asl"
        data, models, best = analyze_outcome(data, outcome, "gaussian", grouping_level)
        plot_data = (
            data if grouping_level == NESTED_GROUPING
            else data.assign(durations=data["average_shot_length"])
        )
        knot_model = models[AUTOMATED_KNOT]
        fig, _ = plot_asl(
            df=plot_data,
            linear_model=models[LINEAR],
            spline_model=knot_model,
            knot=knot_model.knot_found,
            episode_scale=EPISODE_SCALE,
            individual_shots=grouping_level == NESTED_GROUPING,
        )
        try:
            fig.savefig(grouped_figure_path(FIG_SHOT_DURATION_FILE, grouping_level))
        finally:
            plt.close(fig)


if __name__ == "__main__":
    main()
