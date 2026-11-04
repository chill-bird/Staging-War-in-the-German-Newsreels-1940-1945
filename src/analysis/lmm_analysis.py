"""Analyze word-count rates and shot durations with linear mixed models."""

import itertools
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import chi2
import statsmodels.formula.api as smf
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.stats.diagnostic import acorr_ljungbox

from src.analysis.lmm_analysis_plots import (
    plot_word_counts_per_episode_automated_best_knot,
    plot_shot_durations_per_episode_automated_best_knot,
)
from src.util import (
    DOC_DIR,
    KEYFRAME_CLASSIFICATION_TSV,
    get_filtered_chapters_df,
    get_chapter_shot_durations_in_sec,
)

# -----------------------------------------------------------------
# Constants
# -----------------------------------------------------------------

FIG_WORD_COUNTS_FILE = Path(DOC_DIR / "imgs" / "word_counts_lmm.pdf").resolve()
FIG_SHOT_DURATION_FILE = Path(DOC_DIR / "imgs" / "shot_duration_lmm.pdf").resolve()

DF = get_filtered_chapters_df("combat")
KEYFRAMES_DF = pd.read_csv(KEYFRAME_CLASSIFICATION_TSV, sep="\t")
KEYFRAMES_DF["frame"] = KEYFRAMES_DF["frame"].astype(int)
SIMILARITY_THRESHOLD_SHOT_SCALE = 0.165

# Key events
MANUAL_KNOTS = {
    "Barbarossa": {"year": 1941, "episode": 564},
    "Stalingrad Defeat": {"year": 1943, "episode": 648},
}

# -----------------------------------------------------------------
# Core Modeling Functions
# -----------------------------------------------------------------


def _prepare_data(
    df: pd.DataFrame,
    feature: str,
    group_by: str = "episode",
    extra_cols: list[str] | None = None,
) -> pd.DataFrame:
    """Cleans and centers data for modeling, preserving extra columns for variance components."""
    cols = ["episode", feature]
    if group_by not in cols:
        cols.append(group_by)
    if extra_cols:
        for c in extra_cols:
            if c not in cols:
                cols.append(c)

    data = df[cols].copy()
    data["episode"] = pd.to_numeric(data["episode"], errors="coerce")
    data[feature] = pd.to_numeric(data[feature], errors="coerce")
    data = data.replace([np.inf, -np.inf], np.nan).dropna()
    data["episode"] = data["episode"].astype(int)
    if group_by != "episode":
        data = data.dropna(subset=["episode", feature, group_by])

    episode_min = data["episode"].min()
    data["episode_shifted"] = data["episode"] - episode_min
    data = data.reset_index(drop=True)
    return data


def _robust_fit(model, verbose: bool = False):
    """
    Fit a MixedLM using multiple optimizers.

    Warnings are captured rather than printed repeatedly.
    Successful models retain their warning information.
    """

    optimizer_errors = []
    captured_warnings = []

    with warnings.catch_warnings(record=True) as caught:

        # Capture warnings instead of printing them.
        warnings.simplefilter("always")

        for optimizer in ("lbfgs", "powell", "cg"):
            try:
                result = model.fit(
                    reml=False,
                    method=optimizer,
                    maxiter=2000,
                    disp=False,
                )

                variance_components = [
                    result.scale,
                    *np.asarray(result.cov_re).ravel(),
                    *np.asarray(result.vcomp).ravel(),
                ]

                valid = (
                    result.converged
                    and np.isfinite(result.llf)
                    and np.all(np.isfinite(result.fe_params))
                    and np.all(np.isfinite(variance_components))
                    and result.scale > 0
                    and np.all(np.asarray(result.vcomp) >= 0)
                )

                if valid:
                    captured_warnings = [str(w.message) for w in caught]

                    result.fit_warnings = captured_warnings
                    result.optimizer_used = optimizer

                    if verbose and captured_warnings:
                        print(f"{optimizer}: {len(captured_warnings)} warnings")

                    return result

            except Exception as exc:
                optimizer_errors.append(f"{optimizer}: {exc}")

        captured_warnings = [str(w.message) for w in caught]

    if verbose:
        print("All optimizers failed.")
        print("Optimizer errors:", optimizer_errors)
        print("Warnings:", captured_warnings)

    return None


def likelihood_ratio_test(model_simple, model_complex, df_diff):
    lr = 2 * (model_complex.llf - model_simple.llf)
    p_value = chi2.sf(lr, df_diff)
    return {
        "LR statistic": lr,
        "Degrees of freedom": df_diff,
        "p-value": p_value,
    }


def _fit_mixed_model(
    df: pd.DataFrame,
    feature: str,
    formula: str,
    group_by: str,
    vc_formula: dict | None,
    extra_cols: list | None,
):
    data = _prepare_data(df, feature, group_by, extra_cols)
    model = smf.mixedlm(
        formula,
        data=data,
        groups=group_by,
        re_formula="1",
        vc_formula=vc_formula,
    )
    return _robust_fit(model)


def fit_null_model(
    df: pd.DataFrame,
    feature: str,
    group_by: str = "episode",
    vc_formula: dict = None,
    extra_cols: list = None,
):
    """Fit an intercept-only mixed-effects model."""
    return _fit_mixed_model(df, feature, f"{feature} ~ 1", group_by, vc_formula, extra_cols)


def fit_linear_mixed_model(
    df: pd.DataFrame,
    feature: str,
    group_by: str = "episode",
    vc_formula: dict = None,
    extra_cols: list = None,
):
    """Fit a linear mixed-effects model without breakpoints."""
    return _fit_mixed_model(
        df,
        feature,
        f"{feature} ~ episode_shifted",
        group_by,
        vc_formula,
        extra_cols,
    )


def fit_lmm_one_knot(
    df: pd.DataFrame,
    feature: str,
    knot: int,
    group_by: str = "episode",
    vc_formula: dict = None,
    extra_cols: list = None,
):
    data = _prepare_data(df, feature, group_by, extra_cols)
    data["spline_1"] = np.maximum(0, data["episode_shifted"] - (knot - data["episode"].min()))

    model = smf.mixedlm(
        f"{feature} ~ episode_shifted + spline_1",
        data=data,
        groups=group_by,
        re_formula="1",
        vc_formula=vc_formula,
    )
    return _robust_fit(model)


def fit_lmm_two_knots(
    df: pd.DataFrame,
    feature: str,
    knot_1: int,
    knot_2: int,
    group_by: str = "episode",
    vc_formula: dict = None,
    extra_cols: list = None,
):
    data = _prepare_data(df, feature, group_by, extra_cols)
    episode_min = data["episode"].min()
    data["spline_1"] = np.maximum(0, data["episode_shifted"] - (knot_1 - episode_min))
    data["spline_2"] = np.maximum(0, data["episode_shifted"] - (knot_2 - episode_min))

    model = smf.mixedlm(
        f"{feature} ~ episode_shifted + spline_1 + spline_2",
        data=data,
        groups=group_by,
        re_formula="1",
        vc_formula=vc_formula,
    )
    return _robust_fit(model)


def find_best_model_one_knot(
    df: pd.DataFrame,
    feature: str,
    min_episodes_per_segment: int = 15,
    group_by: str = "episode",
    vc_formula: dict = None,
    extra_cols: list = None,
    candidate_episodes: list = None,
):
    """
    Selects the best one-knot LMM using LMM AIC.

    Each candidate breakpoint is fitted using the full dataset.
    """

    data = _prepare_data(df, feature, group_by, extra_cols)

    episodes = np.sort(data["episode"].unique())

    if len(episodes) < 2 * min_episodes_per_segment + 1:
        raise ValueError("Not enough episodes for breakpoint search.")

    # Candidate knots must leave enough episodes on both sides.
    candidates = episodes[min_episodes_per_segment:-min_episodes_per_segment]

    if candidate_episodes is not None:
        candidates = np.intersect1d(candidates, candidate_episodes)

    results = []

    for knot in candidates:
        model = fit_lmm_one_knot(
            df,
            feature,
            int(knot),
            group_by,
            vc_formula,
            extra_cols,
        )

        if model is None:
            continue

        model.knot_found = int(knot)

        results.append(
            {
                "knot": int(knot),
                "AIC": model.aic,
                "BIC": model.bic,
                "log_likelihood": model.llf,
                "model": model,
            }
        )

    if not results:
        raise RuntimeError("No one-knot LMM successfully converged.")

    best = min(results, key=lambda x: x["AIC"])

    results_df = pd.DataFrame(results).drop(columns="model")
    results_df = results_df.sort_values("AIC").reset_index(drop=True)
    results_df["delta_AIC"] = results_df["AIC"] - results_df["AIC"].min()

    print(f"Best one-knot LMM: episode {best['knot']}, AIC = {best['AIC']:.2f}")

    return best["model"], best["knot"], results_df


def find_best_model_two_knots(
    df: pd.DataFrame,
    feature: str,
    min_episodes_per_segment: int = 15,
    group_by: str = "episode",
    vc_formula: dict = None,
    extra_cols: list = None,
    candidate_episodes: list = None,
):
    """
    Selects the best two-knot LMM using LMM AIC.

    All candidate combinations are fitted using the full dataset.
    """

    data = _prepare_data(df, feature, group_by, extra_cols)

    episodes = np.sort(data["episode"].unique())

    if len(episodes) < 3 * min_episodes_per_segment + 1:
        raise ValueError("Not enough episodes for two-breakpoint search.")

    if candidate_episodes is None:
        candidates = episodes[min_episodes_per_segment:-min_episodes_per_segment]
    else:
        candidates = np.intersect1d(episodes, candidate_episodes)

    # Map actual episode IDs to chronological positions.
    positions = {episode: idx for idx, episode in enumerate(episodes)}

    results = []

    for knot_1, knot_2 in itertools.combinations(candidates, 2):
        pos_1 = positions[knot_1]
        pos_2 = positions[knot_2]

        # Ensure enough episodes in all three segments.
        if pos_1 < min_episodes_per_segment:
            continue

        if pos_2 - pos_1 < min_episodes_per_segment:
            continue

        if len(episodes) - pos_2 < min_episodes_per_segment:
            continue

        model = fit_lmm_two_knots(
            df,
            feature,
            int(knot_1),
            int(knot_2),
            group_by,
            vc_formula,
            extra_cols,
        )

        if model is None:
            continue

        model.knots_found = (int(knot_1), int(knot_2))

        results.append(
            {
                "knot_1": int(knot_1),
                "knot_2": int(knot_2),
                "AIC": model.aic,
                "BIC": model.bic,
                "log_likelihood": model.llf,
                "model": model,
            }
        )

    if not results:
        raise RuntimeError("No two-knot LMM successfully converged.")

    best = min(results, key=lambda x: x["AIC"])

    results_df = pd.DataFrame(results).drop(columns="model")
    results_df = results_df.sort_values("AIC").reset_index(drop=True)
    results_df["delta_AIC"] = results_df["AIC"] - results_df["AIC"].min()

    print(f"Best two-knot LMM: {best['knot_1']}, {best['knot_2']}, " f"AIC = {best['AIC']:.2f}")

    return best["model"], (best["knot_1"], best["knot_2"]), results_df


def print_model_statistics(model, title: str):
    if model is None:
        print(f"\n{title}: Model failed to converge.")
        return

    print(f"\n{'=' * 65}")
    print(f"MODEL STATISTICS: {title}")
    print(f"{'=' * 65}")
    print(model.summary())

    var_episode = float(model.cov_re.iloc[0, 0]) if model.cov_re.shape[0] > 0 else 0.0
    var_chapter = float(np.sum(model.vcomp)) if len(model.vcomp) > 0 else 0.0
    var_resid = float(model.scale)
    total_var = var_episode + var_chapter + var_resid
    icc_episode = var_episode / total_var
    icc_chapter = var_chapter / total_var

    print("-" * 65)
    print("Primary Intraclass Correlation Coefficients (ICC):")
    print(f"Episode ICC: {icc_episode:.4f}")
    print(f"Chapter ICC: {icc_chapter:.4f}")
    print(f"Log-Likelihood: {model.llf:.3f}")
    print(f"AIC: {model.aic:.3f}")
    print(f"BIC: {model.bic:.3f}")
    if hasattr(model, "knot_found"):
        print(f"Found Optimal Knot: {model.knot_found}")
    if hasattr(model, "knots_found"):
        print(f"Found Optimal Knots: {model.knots_found}")


def compare_and_select_best_model(models_dict: dict):
    print("\nMODEL COMPARISON (Ranked by AIC)")
    print("-" * 65)

    results = []
    for name, model in models_dict.items():
        if model is not None:
            results.append(
                {
                    "Model Name": name,
                    # "Model": model,
                    "Log-likelihood": model.llf,
                    "AIC": model.aic,
                    "BIC": model.bic,
                }
            )

    if not results:
        print("No valid models to compare.")
        return None

    comparison_df = pd.DataFrame(results).sort_values("AIC", ascending=True).reset_index(drop=True)
    comparison_df["delta_aic"] = comparison_df["AIC"] - comparison_df["AIC"].min()
    comparison_df["delta_bic"] = comparison_df["BIC"] - comparison_df["BIC"].min()
    print(comparison_df.to_string(index=False, float_format="%.3f"))

    best_name = comparison_df.iloc[0]["Model Name"]
    print(f"\nBest model based on AIC: {best_name}")

    return models_dict[best_name]


def _get_model_name(models: dict, selected_model) -> str:
    for name, model in models.items():
        if model is selected_model:
            return name
    raise ValueError("Selected model is not present in the model collection.")


def diagnose_lmm(
    model,
    title="LMM Diagnostics",
    max_lags=20,
):
    """
    Residual diagnostics for a fitted statsmodels MixedLM.

    Examines:
    - Residual distribution
    - Normality
    - Heteroscedasticity patterns
    - Temporal structure
    - Residual autocorrelation
    """

    if model is None:
        raise ValueError("Cannot diagnose a failed model.")

    # Retrieve data used by the fitted model.
    data = model.model.data.frame.copy()

    residuals = np.asarray(model.resid)
    fitted = np.asarray(model.fittedvalues)

    if len(data) != len(residuals):
        raise ValueError("Data and residual lengths do not match.")

    data["residual"] = residuals
    data["fitted"] = fitted

    fig, axes = plt.subplots(2, 3, figsize=(17, 10))

    axes[0, 0].scatter(fitted, residuals, alpha=0.3, s=12)
    axes[0, 0].axhline(0, color="red", linestyle="--")
    axes[0, 0].set_title("Residuals vs fitted")
    axes[0, 0].set_xlabel("Fitted values")
    axes[0, 0].set_ylabel("Residuals")

    stats.probplot(residuals, dist="norm", plot=axes[0, 1])
    axes[0, 1].set_title("Normal Q-Q plot")

    axes[0, 2].hist(residuals, bins=40, density=True, alpha=0.7)
    axes[0, 2].set_title("Residual distribution")
    axes[0, 2].set_xlabel("Residual")

    episode_residuals = data.groupby("episode")["residual"].mean().sort_index()
    axes[1, 0].plot(episode_residuals.index, episode_residuals.values, marker="o", markersize=3)
    axes[1, 0].axhline(0, color="red", linestyle="--")
    axes[1, 0].set_title("Mean residual per episode")
    axes[1, 0].set_xlabel("Episode")
    axes[1, 0].set_ylabel("Mean residual")

    n_episodes = len(episode_residuals)
    if n_episodes >= 4:
        lags = min(max_lags, n_episodes // 2 - 1)
        plot_acf(episode_residuals.values, lags=lags, ax=axes[1, 1], zero=False)
        axes[1, 1].set_title("Episode-level residual ACF")
    else:
        axes[1, 1].text(0.5, 0.5, "Not enough episodes", ha="center", va="center")

    axes[1, 2].scatter(fitted, np.abs(residuals), alpha=0.3, s=12)
    axes[1, 2].set_title("Absolute residuals vs fitted")
    axes[1, 2].set_xlabel("Fitted values")
    axes[1, 2].set_ylabel("|Residuals|")

    fig.suptitle(title, fontsize=15)

    plt.tight_layout()
    plt.show()

    print(f"\nRESIDUAL DIAGNOSTICS: {title}")
    print("-" * 60)
    print("Residual mean:", np.mean(residuals))
    print("Residual standard deviation:", np.std(residuals))
    print("Number of observations:", len(residuals))
    print("Number of episodes:", n_episodes)

    if n_episodes >= 10:
        lb_lags = min(10, n_episodes // 2)
        lb_result = acorr_ljungbox(episode_residuals.values, lags=[lb_lags], return_df=True)
        print("\nLjung-Box test:")
        print(lb_result)

    print("\nRandom-effects covariance:")
    print(model.cov_re)
    print("\nVariance components:")
    print(model.vcomp)
    print("\nResidual variance:")
    print(model.scale)

    return data


# -----------------------------------------------------------------
# Feature Helpers
# -----------------------------------------------------------------


def count_words(s: str | None) -> int | None:
    if s is None:
        return None
    return len(s.split())


def get_duration_sec(row: pd.Series) -> float | None:
    start_ms, end_ms = row["start"], row["end"]
    if pd.notna(start_ms) and pd.notna(end_ms):
        duration_sec = (float(end_ms) - float(start_ms)) / 1000
        return duration_sec if duration_sec > 0 else None
    return None


def words_per_minute(row: pd.Series) -> float | None:
    n_words, duration_sec = row["word_count"], row["duration_sec"]
    if pd.notna(duration_sec) and pd.notna(n_words):
        return n_words * 60 / duration_sec
    return None


def get_key_frames(row: pd.Series) -> pd.DataFrame:
    return KEYFRAMES_DF[
        (KEYFRAMES_DF["filestem"] == row["filestem"])
        & (KEYFRAMES_DF["chapter"] == row["chapter"])
        & (KEYFRAMES_DF["frame"] >= int(row["start_frame"]))
        & (KEYFRAMES_DF["frame"] <= int(row["end_frame"]))
        & (KEYFRAMES_DF["shot_scale_sim_score"] >= SIMILARITY_THRESHOLD_SHOT_SCALE)
    ].copy()


# -----------------------------------------------------------------
# Main Analysis
# -----------------------------------------------------------------


def _prepare_shot_duration_data(df: pd.DataFrame) -> pd.DataFrame:
    """Explode chapter shot durations into one positive-duration row per shot."""
    data = df.copy()
    data["durations"] = data.apply(get_chapter_shot_durations_in_sec, axis=1)
    data["chapter_uid"] = data["filestem"] + "_" + data["chapter"].astype(str)

    assert not data.duplicated(
        ["filestem", "chapter", "episode"]
    ).any(), "Column chapter_uid must uniquely identify chapters for mixed model"
    assert (
        data.groupby("chapter_uid")["episode"].nunique().max() == 1
    ), "One chapter can only belong to 1 episode"

    data = data.explode("durations")
    data["durations"] = pd.to_numeric(data["durations"], errors="coerce")
    data = data.dropna(subset=["durations", "episode"])
    data = data.loc[data["durations"] > 0].copy()
    data["log_duration"] = np.log(data["durations"])
    return data


def _diagnose_selected_model(models: dict, title_prefix: str) -> None:
    best_model = compare_and_select_best_model(models)
    if best_model is None:
        raise RuntimeError("No candidate model successfully converged.")
    model_name = _get_model_name(models, best_model)
    diagnose_lmm(best_model, title=f"{title_prefix}: {model_name}")


def analyze_word_counts() -> None:
    print("\n=================================================")
    print("ANALYSIS: WORD COUNT PER MINUTE")
    print("=================================================")

    df = DF.copy()
    has_transcription = df["audio_transcription"].notna()
    df.loc[has_transcription, "word_count"] = df.loc[
        has_transcription, "audio_transcription"
    ].apply(count_words)
    df["duration_sec"] = df.apply(get_duration_sec, axis=1)
    df["word_count_per_minute"] = df.apply(words_per_minute, axis=1)

    print("\n--- Running LMM Configurations ---")

    m_null = fit_null_model(df, "word_count_per_minute")
    m_linear = fit_linear_mixed_model(df, "word_count_per_minute")
    m_one_knot_barbarossa = fit_lmm_one_knot(
        df, "word_count_per_minute", MANUAL_KNOTS["Barbarossa"]["episode"]
    )
    m_one_knot_stalingrad = fit_lmm_one_knot(
        df, "word_count_per_minute", MANUAL_KNOTS["Stalingrad Defeat"]["episode"]
    )

    print("\n--- Searching for Optimal Breakpoints ---")
    m_one_knot_auto, _, results_one = find_best_model_one_knot(df, "word_count_per_minute")

    models = {
        "Null Model": m_null,
        "Linear (No Knots)": m_linear,
        "One Knot (Manual - Barbarossa)": m_one_knot_barbarossa,
        "One Knot (Manual - Stalingrad)": m_one_knot_stalingrad,
        "One Knot (Automated Best)": m_one_knot_auto,
    }

    for name, model in models.items():
        print_model_statistics(model, name)

    _diagnose_selected_model(models, "WPM")

    fig, _ = plot_word_counts_per_episode_automated_best_knot(
        df, m_linear, m_one_knot_auto, m_one_knot_auto.knot_found
    )
    fig.savefig(FIG_WORD_COUNTS_FILE)

    results_one.to_csv("word_count_one_knot_search.csv", index=False)


def analyze_shot_durations() -> None:
    print("\n\n=================================================")
    print("ANALYSIS: SHOT DURATIONS")
    print("=================================================")

    df = _prepare_shot_duration_data(DF)
    group_by = "episode"
    variance_components = {"chapter_uid": "0 + C(chapter_uid)"}
    extra_cols = ["chapter_uid"]

    print("\n--- Running LMM Configurations ---")

    m_null_shot = fit_null_model(
        df,
        "log_duration",
        group_by,
        vc_formula=variance_components,
        extra_cols=extra_cols,
    )
    m_linear_shot = fit_linear_mixed_model(
        df,
        "log_duration",
        group_by,
        vc_formula=variance_components,
        extra_cols=extra_cols,
    )

    m_one_knot_manual_shot_barbarossa = fit_lmm_one_knot(
        df,
        "log_duration",
        MANUAL_KNOTS["Barbarossa"]["episode"],
        group_by,
        vc_formula=variance_components,
        extra_cols=extra_cols,
    )

    m_one_knot_manual_shot_stalingrad = fit_lmm_one_knot(
        df,
        "log_duration",
        MANUAL_KNOTS["Stalingrad Defeat"]["episode"],
        group_by,
        vc_formula=variance_components,
        extra_cols=extra_cols,
    )

    print("\n--- Searching for Optimal Breakpoints ---")
    m_one_knot_auto_shot, knot_auto_shot, _ = find_best_model_one_knot(
        df,
        "log_duration",
        group_by=group_by,
        vc_formula=variance_components,
        extra_cols=extra_cols,
    )
    # m_two_knots_auto_shot, _, _ = find_best_model_two_knots(
    #     df,
    #     "log_duration",
    #     group_by=group_by,
    #     vc_formula=variance_components,
    #     extra_cols=extra_cols,
    # )

    models_shot = {
        "Null Model": m_null_shot,
        "Linear (No Knots)": m_linear_shot,
        "One Knot (Manual - Barbarossa)": m_one_knot_manual_shot_barbarossa,
        "One Knot (Manual - Stalingrad)": m_one_knot_manual_shot_stalingrad,
        "One Knot (Automated Best)": m_one_knot_auto_shot,
        # "Two Knots (Automated Best)": m_two_knots_auto_shot,
    }

    for name, model in models_shot.items():
        print_model_statistics(model, name)

    _diagnose_selected_model(models_shot, "Shot duration")

    likelihood_test = likelihood_ratio_test(
        m_linear_shot, m_one_knot_manual_shot_stalingrad, df_diff=1
    )
    print("\nLikelihood-ratio test (linear vs. Stalingrad knot):")
    print(likelihood_test)

    fig, _ = plot_shot_durations_per_episode_automated_best_knot(
        df, m_linear_shot, m_one_knot_auto_shot, knot_auto_shot, aggregate_by_chapter=True
    )
    fig.savefig(FIG_SHOT_DURATION_FILE)


def main() -> None:
    # analyze_word_counts()
    analyze_shot_durations()


if __name__ == "__main__":
    main()
