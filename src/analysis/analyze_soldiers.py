"""Analyze chapter soldier-depiction shares or nested keyframes."""

from pathlib import Path

import pandas as pd

from src.analysis.fit_models import (
    AUTOMATED_KNOT,
    EPISODE_SCALE,
    CHAPTER_GROUPING,
    LINEAR,
    analysis_report,
    analyze_outcome,
    classification_observations,
    grouped_figure_path,
)
from src.analysis.plots import plot_depiction_ratio_per_episode
from src.util import DOC_DIR, KEYFRAME_CLASSIFICATION_TSV, get_filtered_chapters_df

SIMILARITY_THRESHOLD_SOLDIERS = 0.17
FIG_SOLDIER_DEPICTIONS_FILE = Path(DOC_DIR / "imgs" / "soldier_depictions_gee.pdf").resolve()
DF_SOLDIERS = get_filtered_chapters_df("soldiers")
KEYFRAMES_DF = pd.read_csv(KEYFRAME_CLASSIFICATION_TSV, sep="\t")


def prepare_soldiers_keyframe_data(
    chapters_df: pd.DataFrame, keyframes_df: pd.DataFrame
) -> pd.DataFrame:
    """Merge chapters and keyframes, keeping episode metadata."""

    chapters = chapters_df.copy()
    keyframes = keyframes_df.copy()
    if chapters[["filestem", "chapter"]].isna().any().any():
        raise ValueError("Chapters must have a filestem and chapter identifier.")

    for col in ["episode", "start_frame", "end_frame"]:
        chapters[col] = pd.to_numeric(chapters[col], errors="coerce")
    for col in ["frame", "content_sim_score"]:
        keyframes[col] = pd.to_numeric(keyframes[col], errors="coerce")

    chapters = chapters.dropna(subset=["episode", "start_frame", "end_frame"]).copy()
    keyframes = keyframes.dropna(subset=["frame", "content_sim_score", "content_class"]).copy()

    chapters["episode"] = chapters["episode"].astype(int)
    keyframes["frame"] = keyframes["frame"].astype(int)

    # Filter for soldier depictions
    keyframes = keyframes[
        keyframes["content_class"].isin(["individual", "multiples", "collective"])
    ].copy()
    # Filter for similarity threshold
    keyframes = keyframes[keyframes["content_sim_score"] >= SIMILARITY_THRESHOLD_SOLDIERS].copy()

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
    retained = prepare_soldiers_keyframe_data(get_filtered_chapters_df("soldiers"), keyframes)
    class_column = "content_class"
    classes = {
        "is_individual": "individual",
        "is_multiple": "multiples",
        "is_collective": "collective",
    }
    print(f"Soldier similarity threshold: {SIMILARITY_THRESHOLD_SOLDIERS}")
    return classification_observations(retained, class_column, classes, grouping_level)


def main(grouping_level: str = CHAPTER_GROUPING) -> None:
    with analysis_report("soldiers", grouping_level):
        observations = prepare_classification_data(grouping_level)
        data, individual_models, _ = analyze_outcome(observations, "is_individual", "binomial", grouping_level)
        _, collective_models, _ = analyze_outcome(observations, "is_collective", "binomial", grouping_level)
        plot = plot_depiction_ratio_per_episode(
            data,
            model_individual_linear=individual_models[LINEAR],
            model_individual_spline=individual_models[AUTOMATED_KNOT],
            model_collective_linear=collective_models[LINEAR],
            model_collective_spline=collective_models[AUTOMATED_KNOT],
            episode_min=int(data["episode"].min()),
            episode_scale=EPISODE_SCALE,
        )
        output = grouped_figure_path(FIG_SOLDIER_DEPICTIONS_FILE, grouping_level)
        output.parent.mkdir(parents=True, exist_ok=True)
        try:
            plot.savefig(output)
        finally:
            plot.close()


if __name__ == "__main__":
    grouping_level = CHAPTER_GROUPING  # Use "nested" for keyframes within chapters and episodes.
    main(grouping_level)
