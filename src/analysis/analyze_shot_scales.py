"""Analyze chapter shot-scale shares or keyframes nested within chapters and episodes."""

from pathlib import Path

import pandas as pd

from src.analysis.fit_models import (
    AUTOMATED_KNOT,
    LINEAR,
    EPISODE_SCALE,
    CHAPTER_GROUPING,
    analysis_report,
    analyze_outcome,
    classification_observations,
    grouped_figure_path,
)
from src.analysis.plots import plot_shot_scale_per_episode
from src.util import DOC_DIR, KEYFRAME_CLASSIFICATION_TSV, get_filtered_chapters_df

SIMILARITY_THRESHOLD_SHOT_SCALE = 0.165
FIG_SHOT_SCALE_FILE = Path(DOC_DIR / "imgs" / "shot_scale_gee.pdf").resolve()
DF_COMBAT = get_filtered_chapters_df("combat")
KEYFRAMES_DF = pd.read_csv(KEYFRAME_CLASSIFICATION_TSV, sep="\t")


def prepare_shot_scale_keyframe_data(
    chapters_df: pd.DataFrame, keyframes_df: pd.DataFrame
) -> pd.DataFrame:
    """Merge chapters and keyframes, keeping episode metadata."""

    chapters = chapters_df.copy()
    keyframes = keyframes_df.copy()
    if chapters[["filestem", "chapter"]].isna().any().any():
        raise ValueError("Chapters must have a filestem and chapter identifier.")

    for col in ["episode", "start_frame", "end_frame"]:
        chapters[col] = pd.to_numeric(chapters[col], errors="coerce")
    for col in ["frame", "shot_scale_sim_score"]:
        keyframes[col] = pd.to_numeric(keyframes[col], errors="coerce")

    chapters = chapters.dropna(subset=["episode", "start_frame", "end_frame"]).copy()
    keyframes = keyframes.dropna(
        subset=["frame", "shot_scale_sim_score", "shot_scale_class"]
    ).copy()

    chapters["episode"] = chapters["episode"].astype(int)
    keyframes["frame"] = keyframes["frame"].astype(int)

    # Filter for similarity threshold
    keyframes = keyframes[
        keyframes["shot_scale_sim_score"] >= SIMILARITY_THRESHOLD_SHOT_SCALE
    ].copy()

    data = keyframes.merge(
        chapters[["filestem", "chapter", "start_frame", "end_frame", "episode", "year"]],
        on=["filestem", "chapter"],
        how="inner",
        validate="many_to_one",
    )

    data = data[
        data["frame"].between(data["start_frame"], data["end_frame"], inclusive="both")
    ].copy()

    # Define standard clusters and scaling
    data["episode_id"] = data["episode"].astype(str)
    data["chapter_uid"] = data["filestem"].astype(str) + "_" + data["chapter"].astype(str)

    if not data.empty:
        episode_min = int(data["episode"].min())
        data["episode_scaled"] = (data["episode"] - episode_min) / EPISODE_SCALE

    return data


def prepare_classification_data(grouping_level: str = CHAPTER_GROUPING) -> pd.DataFrame:
    keyframes = pd.read_csv(KEYFRAME_CLASSIFICATION_TSV, sep="\t")
    retained = prepare_shot_scale_keyframe_data(get_filtered_chapters_df("combat"), keyframes)
    class_column = "shot_scale_class"
    classes = {"is_close": "Close-up", "is_medium": "Medium Shot", "is_long": "Long Shot"}
    print(f"Shot-scale similarity threshold: {SIMILARITY_THRESHOLD_SHOT_SCALE}")
    return classification_observations(retained, class_column, classes, grouping_level)


def main(grouping_level: str = CHAPTER_GROUPING) -> None:
    with analysis_report("shot_scales", grouping_level):
        observations = prepare_classification_data(grouping_level)
        data, close_models, _ = analyze_outcome(observations, "is_close", "binomial", grouping_level)
        _, long_models, _ = analyze_outcome(observations, "is_long", "binomial", grouping_level)
        plot = plot_shot_scale_per_episode(
            data,
            model_close_linear=close_models[LINEAR],
            model_close_spline=close_models[AUTOMATED_KNOT],
            model_long_linear=long_models[LINEAR],
            model_long_spline=long_models[AUTOMATED_KNOT],
            episode_min=int(data["episode"].min()),
            episode_scale=EPISODE_SCALE,
        )
        output = grouped_figure_path(FIG_SHOT_SCALE_FILE, grouping_level)
        output.parent.mkdir(parents=True, exist_ok=True)
        try:
            plot.savefig(output)
        finally:
            plot.close()


if __name__ == "__main__":
    grouping_level = CHAPTER_GROUPING  # Use "nested" for keyframes within chapters and episodes.
    main(grouping_level)
