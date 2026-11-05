"""
gee_binary_analysis

===

Generalized Estimating Equations (GEE) Logistic Models for binary classifications:
- shot scale (close shots vs rest)
- soldier depictions (individual vs collective)
"""

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf

from src.analysis.gee_analysis_plots import (
    plot_depiction_ratio_per_episode,
    plot_shot_scale_per_episode,
)
from src.util import (
    DOC_DIR,
    KEYFRAME_CLASSIFICATION_TSV,
    get_filtered_chapters_df,
)

# -----------------------------------------------------------------
# Constants
# -----------------------------------------------------------------

FIG_SOLDIER_DEPICTIONS_FILE = Path(DOC_DIR / "imgs" / "soldier_depictions_gee.pdf").resolve()
FIG_SHOT_SCALE_FILE = Path(DOC_DIR / "imgs" / "shot_scale_gee.pdf").resolve()
BARBAROSSA_MODEL = "One Knot (Manual - Barbarossa)"
KNOT_SEARCH_RESULTS_DIR = Path(Path(__file__).parent.resolve() / "knot_search_results").resolve()

DF_COMBAT = get_filtered_chapters_df("combat")
DF_SOLDIERS = get_filtered_chapters_df("soldiers")
KEYFRAMES_DF = pd.read_csv(KEYFRAME_CLASSIFICATION_TSV, sep="\t")

SIMILARITY_THRESHOLD_SHOT_SCALE = 0.165
SIMILARITY_THRESHOLD_SOLDIERS = 0.17

EPISODE_SCALE = 10.0

# Key events
MANUAL_KNOTS = {
    "Barbarossa": {"year": 1941, "episode": 564},
    "Stalingrad Defeat": {"year": 1943, "episode": 648},
}

# -----------------------------------------------------------------
# Core Modeling Functions
# -----------------------------------------------------------------


def calculate_baseline_iccs(data: pd.DataFrame, target_col: str):
    """
    GEE models cluster by one group. This function runs two null models
    to output the working correlation (ICC equivalent) for both levels.
    """
    print("\n--- Baseline Working Correlations (GEE 'ICC' Equivalent) ---")
    try:
        model_ep = smf.gee(
            f"{target_col} ~ 1",
            groups=data["episode_id"],
            data=data,
            family=sm.families.Binomial(),
            cov_struct=sm.cov_struct.Exchangeable(),
        ).fit()
        print(f"Episode-level Working Correlation: {model_ep.cov_struct.dep_params:.4f}")
    except Exception as e:
        print(f"Episode-level ICC failed: {e}")

    try:
        model_ch = smf.gee(
            f"{target_col} ~ 1",
            groups=data["chapter_uid"],
            data=data,
            family=sm.families.Binomial(),
            cov_struct=sm.cov_struct.Exchangeable(),
        ).fit()
        print(f"Chapter-level Working Correlation: {model_ch.cov_struct.dep_params:.4f}")
    except Exception as e:
        print(f"Chapter-level ICC failed: {e}")


def fit_gee_model(data: pd.DataFrame, formula: str, group_by: str):
    """Wrapper to fit a GEE model and return None if it fails to converge."""
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = smf.gee(
                formula=formula,
                groups=data[group_by],
                data=data,
                family=sm.families.Binomial(),
                cov_struct=sm.cov_struct.Exchangeable(),
            )
            result = model.fit()
            return result
    except Exception as e:
        # print(f"Model failed: {formula} | Error: {e}")
        return None


def fit_null_model(data: pd.DataFrame, target_col: str, group_by: str):
    return fit_gee_model(data, f"{target_col} ~ 1", group_by)


def fit_linear_model(data: pd.DataFrame, target_col: str, group_by: str):
    return fit_gee_model(data, f"{target_col} ~ episode_scaled", group_by)


def fit_quadratic_model(data: pd.DataFrame, target_col: str, group_by: str):
    return fit_gee_model(data, f"{target_col} ~ episode_scaled + I(episode_scaled**2)", group_by)


def fit_one_knot_model(data: pd.DataFrame, target_col: str, knot_episode: int, group_by: str):
    df = data.copy()
    episode_min = df["episode"].min()
    knot_scaled = (knot_episode - episode_min) / EPISODE_SCALE

    df["spline_1"] = np.maximum(0, df["episode_scaled"] - knot_scaled)
    model = fit_gee_model(df, f"{target_col} ~ episode_scaled + spline_1", group_by)

    if model is not None:
        model.knot_found = knot_episode
    return model


def find_best_model_one_knot(
    data: pd.DataFrame, target_col: str, group_by: str, min_episodes_per_segment: int = 15
):
    """Selects the best one-knot GEE using QIC."""
    episodes = np.sort(data["episode"].unique())
    candidates = episodes[min_episodes_per_segment:-min_episodes_per_segment]

    results = []

    for knot in candidates:
        model = fit_one_knot_model(data, target_col, int(knot), group_by)
        if model is not None:
            results.append({"knot": int(knot), "QIC": model.qic(scale=1.0)[0], "model": model})

    if not results:
        print("No one-knot GEE successfully converged.")
        return None, None, None

    best = min(results, key=lambda x: x["QIC"])

    results_df = pd.DataFrame(results).drop(columns="model")
    results_df = results_df.sort_values("QIC").reset_index(drop=True)
    results_df["delta_QIC"] = results_df["QIC"] - results_df["QIC"].min()

    print(f"Best one-knot GEE: episode {best['knot']}, QIC = {best['QIC']:.2f}")

    return best["model"], best["knot"], results_df


def print_model_statistics(model, title: str):
    if model is None:
        print(f"\n{title}: Model failed to converge.")
        return

    print(f"\n{'=' * 65}")
    print(f"MODEL STATISTICS: {title}")
    print(f"{'=' * 65}")
    print(model.summary())

    print("-" * 65)
    print("ODDS RATIOS:")
    print(np.exp(model.params))

    print("-" * 65)
    print(f"QIC (Model Fit Criterion): {model.qic(scale=1.0)[0]:.3f}")
    if hasattr(model, "knot_found"):
        print(f"Found Optimal Knot: {model.knot_found}")

    # Working correlation (GEE ICC)
    if hasattr(model.cov_struct, "dep_params"):
        print(f"Working Correlation (Alpha): {model.cov_struct.dep_params:.4f}")


def compare_and_select_best_model(models_dict: dict):
    print("\nMODEL COMPARISON (Ranked by QIC)")
    print("-" * 65)

    results = []
    for name, model in models_dict.items():
        if model is not None:
            results.append(
                {
                    "Model Name": name,
                    "QIC": model.qic(scale=1.0)[0],
                }
            )

    if not results:
        print("No valid models to compare.")
        return None

    comparison_df = pd.DataFrame(results).sort_values("QIC", ascending=True).reset_index(drop=True)
    comparison_df["delta_QIC"] = comparison_df["QIC"] - comparison_df["QIC"].min()
    print(comparison_df.to_string(index=False, float_format="%.3f"))

    best_name = comparison_df.iloc[0]["Model Name"]
    print(f"\nBest model based on QIC: {best_name}")

    return models_dict[best_name]


# -----------------------------------------------------------------
# Feature Helpers & Data Prep
# -----------------------------------------------------------------


def prepare_shot_scale_keyframe_data(
    chapters_df: pd.DataFrame, keyframes_df: pd.DataFrame
) -> pd.DataFrame:
    """Merge chapters and keyframes, keeping episode metadata."""

    chapters = chapters_df.copy()
    keyframes = keyframes_df.copy()

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
    )

    data = data[
        data["frame"].between(data["start_frame"], data["end_frame"], inclusive="both")
    ].copy()

    # Define standard clusters and scaling
    data["episode_id"] = data["episode"].astype(str)
    data["chapter_uid"] = data["filestem"].astype(str) + "_" + data["chapter"].astype(str)

    episode_min = int(data["episode"].min())
    data["episode_scaled"] = (data["episode"] - episode_min) / EPISODE_SCALE

    return data


# TODO
def prepare_soldiers_keyframe_data(
    chapters_df: pd.DataFrame, keyframes_df: pd.DataFrame
) -> pd.DataFrame:
    """Merge chapters and keyframes, keeping episode metadata."""

    chapters = chapters_df.copy()
    keyframes = keyframes_df.copy()

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
    )

    data = data[
        data["frame"].between(data["start_frame"], data["end_frame"], inclusive="both")
    ].copy()

    # Define standard clusters and scaling
    data["episode_id"] = data["episode"].astype(str)
    data["chapter_uid"] = data["filestem"].astype(str) + "_" + data["chapter"].astype(str)

    episode_min = int(data["episode"].min())
    data["episode_scaled"] = (data["episode"] - episode_min) / EPISODE_SCALE

    return data


# -----------------------------------------------------------------
# Main Analysis Wrappers
# -----------------------------------------------------------------


def analyze_binary_classification(
    data: pd.DataFrame, target_col: str, analysis_title: str, group_by: str = "chapter_uid"
):
    """
    Main orchestration function to run the full suite of GEE models on a binary outcome.
    """
    print("\n" + "=" * 65)
    print(f"ANALYSIS: {analysis_title.upper()}")
    print("=" * 65)

    # 1. Output Baseline ICCs (Working Correlations)
    calculate_baseline_iccs(data, target_col)

    print("\n--- Running GEE Configurations ---")

    # 2. Fit standard models
    m_null = fit_null_model(data, target_col, group_by)
    m_linear = fit_linear_model(data, target_col, group_by)
    m_quadratic = fit_quadratic_model(data, target_col, group_by)

    # 3. Fit Manual Knots
    m_knot_barbarossa = fit_one_knot_model(
        data, target_col, MANUAL_KNOTS["Barbarossa"]["episode"], group_by
    )
    m_knot_stalingrad = fit_one_knot_model(
        data, target_col, MANUAL_KNOTS["Stalingrad Defeat"]["episode"], group_by
    )

    # 4. Search for Optimal Knot
    print("\n--- Searching for Optimal Breakpoint ---")
    m_knot_auto, knot, results_df = find_best_model_one_knot(data, target_col, group_by)

    # 5. Summarize
    models = {
        "Null Model": m_null,
        "Linear (No Knots)": m_linear,
        "Quadratic": m_quadratic,
        "One Knot (Manual - Barbarossa)": m_knot_barbarossa,
        "One Knot (Manual - Stalingrad)": m_knot_stalingrad,
        "One Knot (Automated Best)": m_knot_auto,
    }

    for name, model in models.items():
        print_model_statistics(model, name)

    compare_and_select_best_model(models)

    if results_df is not None:
        filename = Path(KNOT_SEARCH_RESULTS_DIR / f"{target_col}_knot_search_results.csv").resolve()
        results_df.to_csv(filename, index=False)
        print(f"\nSaved knot search results to {filename}")

    return models


# -----------------------------------------------------------------
# Specific Analysis Executions
# -----------------------------------------------------------------


def analyze_shot_scale():
    # 1. Prepare data
    data = prepare_shot_scale_keyframe_data(DF_COMBAT, KEYFRAMES_DF)

    # Close Shots
    data["is_close"] = (data["shot_scale_class"] == "Close-up").astype(int)
    models_close = analyze_binary_classification(
        data=data,
        target_col="is_close",
        analysis_title="Shot Scale (Close-Ups vs Rest)",
        group_by="chapter_uid",
    )

    # Long Shots
    data["is_long"] = (data["shot_scale_class"] == "Long Shot").astype(int)
    models_long = analyze_binary_classification(
        data=data,
        target_col="is_long",
        analysis_title="Shot Scale (Long Shots vs Rest)",
        group_by="chapter_uid",
    )

    plot = plot_shot_scale_per_episode(
        data,
        model_close=models_close[BARBAROSSA_MODEL],
        model_long=models_long[BARBAROSSA_MODEL],
        barbarossa_episode=MANUAL_KNOTS["Barbarossa"]["episode"],
        episode_min=int(data["episode"].min()),
        episode_scale=EPISODE_SCALE,
    )
    plot.savefig(FIG_SHOT_SCALE_FILE)
    plot.close()


def analyze_soldier_depictions():
    """
    Example of how to adjust the script for your other classification.
    """
    data = prepare_soldiers_keyframe_data(DF_SOLDIERS, KEYFRAMES_DF)

    # Individuals
    data["is_individual"] = (data["content_class"] == "individual").astype(int)
    models_individual = analyze_binary_classification(
        data=data,
        target_col="is_individual",
        analysis_title="Soldier Classification (Individual vs Other Soldier Depiction)",
        group_by="chapter_uid",
    )

    # Collective
    data["is_collective"] = (data["content_class"] == "collective").astype(int)
    models_collective = analyze_binary_classification(
        data=data,
        target_col="is_collective",
        analysis_title="Soldier Classification (Collective vs Other Soldier Depiction)",
        group_by="chapter_uid",
    )

    plot = plot_depiction_ratio_per_episode(
        data,
        model_individual_spline=models_individual["One Knot (Automated Best)"],
        model_individual_quadratic=models_individual["Quadratic"],
        model_collective=models_collective[BARBAROSSA_MODEL],
        barbarossa_episode=MANUAL_KNOTS["Barbarossa"]["episode"],
        episode_min=int(data["episode"].min()),
        episode_scale=EPISODE_SCALE,
    )
    plot.savefig(FIG_SOLDIER_DEPICTIONS_FILE)
    plot.close()


def main() -> None:
    analyze_shot_scale()
    # analyze_soldier_depictions()


if __name__ == "__main__":
    main()
