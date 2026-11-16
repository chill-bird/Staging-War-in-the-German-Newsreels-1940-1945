"""GEE models for chapters or keyframes nested in chapters and episodes."""

import sys
import traceback
from contextlib import contextmanager, redirect_stderr, redirect_stdout

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf

from src.analysis.plots import RESIDUALS_DIR, plot_residual_qq
from src.util import ANALYSIS_DIR

CHAPTER_GROUPING = "episode"
NESTED_GROUPING = "nested"
EPISODE_SCALE = 10.0
MANUAL_KNOTS = {"Barbarossa": 564, "Stalingrad": 648}
LINEAR = "Linear"
QUADRATIC = "Quadratic"
BARBAROSSA = "Barbarossa"
STALINGRAD = "Stalingrad"
GROUPING_LEVELS = (CHAPTER_GROUPING, NESTED_GROUPING)


def validate_grouping(grouping_level: str) -> None:
    if grouping_level not in GROUPING_LEVELS:
        raise ValueError(f"Unknown grouping level: {grouping_level!r}.")


def grouped_figure_path(path, grouping_level: str):
    validate_grouping(grouping_level)
    return path.with_name(f"{path.stem}_{grouping_level}{path.suffix}")


def classification_observations(
    retained: pd.DataFrame, class_column: str, classes: dict[str, str], grouping_level: str
) -> pd.DataFrame:
    """Prepare chapter shares for episode mode or individual nested keyframes."""
    validate_grouping(grouping_level)
    if retained.empty:
        raise ValueError("No retained keyframes.")
    if not retained[class_column].isin(classes.values()).all():
        raise ValueError(f"Unrecognized classes: {retained[class_column].unique()}")
    if retained[["chapter_uid", "episode", "year"]].isna().any().any():
        raise ValueError("Retained keyframes must identify their chapter, episode and year.")
    if (retained.groupby("chapter_uid")[["episode", "year"]].nunique() != 1).any().any():
        raise ValueError("Each chapter must belong to exactly one episode and year.")
    if retained.duplicated(["filestem", "chapter", "frame"]).any():
        raise ValueError("Retained keyframes must be unique within each chapter.")
    retained = retained.copy()
    for column, label in classes.items():
        retained[column] = (retained[class_column] == label).astype(int)
    print(f"Retained {len(retained)} keyframes in {retained['chapter_uid'].nunique()} chapters.")
    if grouping_level == NESTED_GROUPING:
        print("Each retained keyframe is an observation nested in its chapter and episode.")
        return retained
    aggregations = {"episode": ("episode", "first"), "year": ("year", "first")}
    aggregations.update({column: (column, "mean") for column in classes})
    chapters = retained.groupby("chapter_uid", as_index=False).agg(**aggregations)
    print(
        "Chapters without retained keyframes are excluded; each retained chapter has equal weight."
    )
    return chapters


def prepare_model_data(
    data: pd.DataFrame, outcome: str, grouping_level: str = CHAPTER_GROUPING
) -> pd.DataFrame:
    """Keep valid chapter, keyframe or shot observations and center episode time."""
    validate_grouping(grouping_level)
    data = data.copy()
    if data["chapter_uid"].isna().any():
        raise ValueError("All observations must identify their chapter.")
    if grouping_level == CHAPTER_GROUPING and data["chapter_uid"].duplicated().any():
        raise ValueError("Analysis requires one observation per identified chapter.")
    if grouping_level == NESTED_GROUPING and outcome in ("log_asl", "log_wpm"):
        raise ValueError("Chapter-average ASL and WPM support chapter observations only.")
    for column in ("episode", outcome):
        data[column] = pd.to_numeric(data[column], errors="coerce")
    valid = np.isfinite(data["episode"]) & np.isfinite(data[outcome])
    unit = observation_unit(outcome, grouping_level)
    print(f"Excluded {int((~valid).sum())} {unit} with missing or non-finite model values.")
    data = data.loc[valid].copy().reset_index(drop=True)
    print(f"Regression coefficients are scaled to {int(EPISODE_SCALE)}-episode blocks.")
    if data.empty:
        raise ValueError(f"No valid observations for {outcome}.")
    if (data.groupby("chapter_uid")["episode"].nunique() != 1).any():
        raise ValueError("Each chapter must belong to exactly one episode.")
    if grouping_level == CHAPTER_GROUPING:
        if "year" in data and (data.groupby("episode")["year"].nunique(dropna=False) != 1).any():
            raise ValueError("Each episode must belong to exactly one year.")
        raw_outcomes = {"log_wpm": "word_count_per_minute", "log_asl": "average_shot_length"}
        raw_outcome = raw_outcomes.get(outcome)
        if raw_outcome:
            data[raw_outcome] = pd.to_numeric(data[raw_outcome], errors="coerce")
            if not (np.isfinite(data[raw_outcome]) & (data[raw_outcome] > 0)).all():
                raise ValueError(f"{raw_outcome} must be finite and positive for valid chapters.")
    data["episode_scaled"] = (data["episode"] - data["episode"].min()) / EPISODE_SCALE
    return data


def fit_gee_model(
    data: pd.DataFrame,
    outcome: str,
    terms: str,
    family: str,
    grouping_level: str = CHAPTER_GROUPING,
):
    """Fit an episode-clustered GEE with the selected observation level."""
    validate_grouping(grouping_level)
    if grouping_level == CHAPTER_GROUPING:
        if data["chapter_uid"].isna().any() or data["chapter_uid"].duplicated().any():
            raise ValueError("Chapter grouping requires one observation per identified chapter.")
        if (data.groupby("chapter_uid")["episode"].nunique() != 1).any():
            raise ValueError("Each chapter must belong to exactly one episode.")
    if family == "gaussian":
        distribution = sm.families.Gaussian()
    elif family == "binomial":
        distribution = sm.families.Binomial()
        if not data[outcome].between(0, 1).all():
            raise ValueError("Binomial proportions must be between 0 and 1.")
        if grouping_level == NESTED_GROUPING and not data[outcome].isin([0, 1]).all():
            raise ValueError("Nested binomial outcomes must be individual keyframe indicators.")
    else:
        raise ValueError(f"Unknown GEE family: {family}")
    if data[["episode", "episode_scaled", outcome]].isna().any().any():
        raise ValueError("GEE data must not contain missing model values.")
    if grouping_level == NESTED_GROUPING:
        if data["chapter_uid"].isna().any():
            raise ValueError("Nested GEE requires a chapter identifier for every observation.")
        if (data.groupby("chapter_uid")["episode"].nunique() != 1).any():
            raise ValueError("Each chapter must belong to exactly one episode.")
        chapter_counts = data.groupby("episode")["chapter_uid"].nunique()
        observation_counts = data.groupby("chapter_uid").size()
        if not (chapter_counts > 1).any() or not (observation_counts > 1).any():
            raise ValueError(
                "Nested GEE requires pairs within chapters and between chapters within episodes."
            )
        data = data.copy()
        data["chapter_cluster_id"] = pd.factorize(data["chapter_uid"])[0]
    formula = f"{outcome} ~ {terms}"
    model = smf.gee(
        formula,
        groups="episode",
        data=data,
        family=distribution,
        cov_struct=(
            sm.cov_struct.Exchangeable()
            if grouping_level == CHAPTER_GROUPING
            else sm.cov_struct.Nested()
        ),
        dep_data="0 + chapter_cluster_id" if grouping_level == NESTED_GROUPING else None,
        missing="raise",
    )
    if np.linalg.matrix_rank(model.exog) != model.exog.shape[1]:
        raise ValueError(f"Model design is rank deficient: {formula}")
    result = model.fit()
    if not result.converged or not np.isfinite(result.params).all():
        raise RuntimeError(f"GEE failed to converge: {formula}")
    if grouping_level == NESTED_GROUPING:
        structure = result.cov_struct
        if (
            not np.isfinite(structure.scale)
            or structure.scale <= 0
            or not np.isfinite(structure.vcomp_coeff).all()
        ):
            raise RuntimeError(f"Nested GEE produced invalid dependence estimates: {formula}")
    return result


def fit_one_knot_model(
    data: pd.DataFrame, outcome: str, knot: int, family: str, grouping_level: str = CHAPTER_GROUPING
):
    data = data.copy()
    knot_scaled = (knot - data["episode"].min()) / EPISODE_SCALE
    data["spline_1"] = np.maximum(0, data["episode_scaled"] - knot_scaled)
    model = fit_gee_model(data, outcome, "episode_scaled + spline_1", family, grouping_level)
    model.knot_found = knot
    return model


def fit_models(
    data: pd.DataFrame, outcome: str, family: str, grouping_level: str = CHAPTER_GROUPING
) -> dict:
    """Fit the four prespecified time trends on the same observations."""
    return {
        LINEAR: fit_gee_model(data, outcome, "episode_scaled", family, grouping_level),
        QUADRATIC: fit_gee_model(
            data, outcome, "episode_scaled + I(episode_scaled ** 2)", family, grouping_level
        ),
        BARBAROSSA: fit_one_knot_model(
            data, outcome, MANUAL_KNOTS[BARBAROSSA], family, grouping_level
        ),
        STALINGRAD: fit_one_knot_model(
            data, outcome, MANUAL_KNOTS[STALINGRAD], family, grouping_level
        ),
    }


def model_qic(model, scale: float) -> float:
    qic = float(model.qic(scale=scale)[0])
    if not np.isfinite(qic):
        raise ValueError("GEE QIC is not finite.")
    return qic


def find_best_model_one_knot(
    data: pd.DataFrame,
    outcome: str,
    family: str,
    scale: float,
    min_episodes_per_segment: int = 15,
    grouping_level: str = CHAPTER_GROUPING,
):
    """Search observed knots, retaining at least the requested episodes on each side."""
    if min_episodes_per_segment < 1:
        raise ValueError("min_episodes_per_segment must be positive.")
    episodes = np.sort(data["episode"].unique())
    candidates = episodes[min_episodes_per_segment:-min_episodes_per_segment]
    if len(candidates) == 0:
        raise ValueError("Not enough distinct episodes for the requested knot search.")
    results = []
    for knot in candidates:
        try:
            model = fit_one_knot_model(data, outcome, int(knot), family, grouping_level)
            qic = model_qic(model, scale)
        except (ValueError, RuntimeError, np.linalg.LinAlgError) as exc:
            print(f"Skipping knot {knot}: {exc}")
            continue
        results.append({"knot": int(knot), "QIC": qic, "model": model})
    if not results:
        raise RuntimeError("No candidate knot produced a valid converged model.")
    best = min(results, key=lambda row: row["QIC"])
    table = pd.DataFrame(results).drop(columns="model").sort_values("QIC").reset_index(drop=True)
    table["delta_QIC"] = table["QIC"] - table["QIC"].min()
    return best["model"], best["knot"], table


class _Tee:
    def __init__(self, stream, report):
        self.stream = stream
        self.report = report

    def write(self, text):
        self.stream.write(text)
        self.report.write(text)
        self.report.flush()

    def flush(self):
        self.stream.flush()
        self.report.flush()


@contextmanager
def analysis_report(name: str, grouping_level: str = CHAPTER_GROUPING):
    """Mirror stdout, stderr, warnings and failures to the analysis report."""
    validate_grouping(grouping_level)
    path = (
        ANALYSIS_DIR / "reports" / f"results_{name}_gee.txt"
        if grouping_level == CHAPTER_GROUPING
        else ANALYSIS_DIR / "reports" / "nested" / f"results_{name}_gee.txt"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as report:
        with redirect_stdout(_Tee(sys.stdout, report)), redirect_stderr(_Tee(sys.stderr, report)):
            try:
                yield
            except Exception:
                traceback.print_exc()
                raise


def observation_unit(outcome: str, grouping_level: str) -> str:
    if grouping_level == CHAPTER_GROUPING:
        return "chapters"
    return "shots" if outcome == "log_shot_length" else "keyframes"


def analyze_outcome(
    data: pd.DataFrame, outcome: str, family: str, grouping_level: str = CHAPTER_GROUPING
):
    """Fit and report the four prespecified models."""
    data = prepare_model_data(data, outcome, grouping_level)
    print(f"\n{'=' * 65}\nOUTCOME: {outcome}\n{'=' * 65}")
    unit = observation_unit(outcome, grouping_level)
    print(
        f"Grouping: {grouping_level}. Observations: {len(data)} {unit} in {data['episode'].nunique()} episodes."
    )
    chapters_per_episode = data.groupby("episode")["chapter_uid"].nunique()
    if grouping_level == NESTED_GROUPING:
        observations_per_chapter = data.groupby("chapter_uid").size()
        print(f"Chapters: {len(observations_per_chapter)}.")
        print(
            f"{unit.capitalize()} per chapter (min/median/max): {observations_per_chapter.min()}/"
            f"{observations_per_chapter.median():g}/{observations_per_chapter.max()}."
        )
    print(
        f"Chapters per episode (min/median/max): {chapters_per_episode.min()}/"
        f"{chapters_per_episode.median():g}/{chapters_per_episode.max()}."
    )
    print(
        "GEE: "
        + (
            "exchangeable working correlation for chapters within episodes"
            if grouping_level == CHAPTER_GROUPING
            else "nested working correlation for chapters within episodes"
        )
        + "; robust standard errors clustered by episode."
    )
    if family == "binomial":
        print(
            "Binomial logit GEE on "
            + (
                "equally weighted chapter proportions (fractional response)."
                if grouping_level == CHAPTER_GROUPING
                else "individual keyframe classifications (binary response)."
            )
        )
    else:
        print(f"Gaussian identity-link GEE on the log-transformed outcome for {unit}.")
    models = fit_models(data, outcome, family, grouping_level)
    scale = 1.0 if family == "binomial" else float(models[LINEAR].scale)
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("The common QIC scale must be finite and positive.")
    scores = {name: model_qic(model, scale) for name, model in models.items()}
    for name, model in models.items():
        print(f"\nMODEL: {name}\n{model.summary()}")
        print(f"QIC (common scale {scale:.6g}): {scores[name]:.3f}")
        if grouping_level == CHAPTER_GROUPING:
            print(
                f"Estimated exchangeable working correlation between chapters "
                f"within an episode: {model.cov_struct.dep_params:.4f}"
            )
        else:
            structure = model.cov_struct
            episode_var, chapter_var = structure.vcomp_coeff
            print(
                f"Working variance components: episode={episode_var:.6g}, "
                f"chapter within episode={chapter_var:.6g}, total scale={structure.scale:.6g}."
            )
            print(
                f"Working correlation, same episode but different chapters: {episode_var / structure.scale:.4f}"
            )
            print(
                f"Working correlation, same chapter within an episode: {(episode_var + chapter_var) / structure.scale:.4f}"
            )
    best = min(scores, key=scores.get)
    print("\nPRESPECIFIED MODEL COMPARISON (QIC)")
    for name in sorted(scores, key=scores.get):
        print(f"{name}:\t\t{scores[name]:.3f};\t\tdelta QIC: {scores[name] - scores[best]:.3f}")
    print(f"Best prespecified model: {best}")

    selected = models[best]
    RESIDUALS_DIR.mkdir(parents=True, exist_ok=True)
    residuals = np.asarray(
        selected.resid_pearson if family == "binomial" else selected.resid_response
    )
    print(
        "Residual diagnostics: marginal Pearson residuals for proportions"
        if family == "binomial"
        else "Residual diagnostics: marginal residuals on the log scale"
    )
    fig, _ = plot_residual_qq(residuals, title=f"{outcome}\n{best}", ylabel="Marginal residuals")
    path = RESIDUALS_DIR / f"{outcome}_{grouping_level}_gee_qq.pdf"
    try:
        fig.savefig(path)
    finally:
        plt.close(fig)

    return data, models, best
