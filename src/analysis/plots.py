"""
lmm_analysis_plot.py
---

Plots for lmm_analysis
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import font_manager
from matplotlib.lines import Line2D
from PIL import Image
from statsmodels.graphics.gofplots import qqplot
from tueplots import bundles

# from tueplots import figsizes
from src.util import DOC_DIR, get_keyframe_path

# Register Noto Sans font
NOTO_SANS_DIR = Path(__file__).parent / "Noto_Sans"
for ttf_file in NOTO_SANS_DIR.glob("*.ttf"):
    font_manager.fontManager.addfont(ttf_file)
subdir = NOTO_SANS_DIR / "static"
if subdir.is_dir():
    for ttf_file in subdir.glob("*.ttf"):
        font_manager.fontManager.addfont(ttf_file)

plt.rcParams.update(bundles.beamer_moml())
plt.rcParams.update(
    {
        "figure.dpi": 200,
        # "figure.figsize": (5.5324802778 / 1.5, 8.6938975 / 3*1.5),
        "font.family": "Noto Sans Condensed",
        "font.weight": 500,  # Medium weight globally
        "axes.titlesize": 14,  # title size
        "axes.titleweight": 500,  # Medium weight for titles
        "axes.labelsize": 12,  # x and y label size
        "axes.labelweight": 500,  # Medium weight for axis labels
        "xtick.labelsize": 10,  # x tick labels
        "ytick.labelsize": 10,  # y tick labels
        "legend.fontsize": 10,  # legend text
        "legend.labelspacing": 0,  # legend spacing
    }
)
# Color palette
years = [1940, 1941, 1942, 1943, 1944, 1945]
palette = dict(zip(years, sns.color_palette("tab20", len(years))))
palette[1945] = "gold"  # Override 1945
RED = "#d7191c"
BLUE = "#2b83ba"

RESIDUALS_DIR = DOC_DIR / "imgs" / "residuals"



def plot_wpm(
    df: pd.DataFrame, linear_model, spline_model, knot: int | None, episode_scale: float
):
    """Plot observed WPM with back-transformed log-WPM GEE or LMM predictions."""

    plot_piecewise = _has_piecewise_model(spline_model, knot)
    sort_cols = ["episode"] + (["chapter"] if "chapter" in df else [])
    plot_df = df.sort_values(sort_cols).reset_index(drop=True)
    model_df = linear_model.model.data.frame
    episode_min = model_df["episode"].min()

    x_grid = np.linspace(
        plot_df["episode"].min(),
        plot_df["episode"].max(),
        500,
    )
    x_scaled = (x_grid - episode_min) / episode_scale

    linear_params = _fixed_params(linear_model)
    linear_slope_name = _episode_slope_name(linear_params.index)
    linear_design = pd.DataFrame(
        {
            "Intercept": np.ones_like(x_grid),
            linear_slope_name: x_scaled,
        }
    )
    y_linear, _, _ = _model_prediction_ci(linear_model, linear_design)

    if plot_piecewise:
        spline_params = _fixed_params(spline_model)
        spline_slope_name = _episode_slope_name(spline_params.index)
        # Post-breakpoint slope
        knot_scaled = (knot - episode_min) / episode_scale
        spline_1 = np.maximum(x_scaled - knot_scaled, 0)
        spline_design = pd.DataFrame(
            {
                "Intercept": np.ones_like(x_grid),
                spline_slope_name: x_scaled,
                "spline_1": spline_1,
            }
        )
        y_spline, spline_lo, spline_hi = _model_prediction_ci(spline_model, spline_design)

    # Back-transform the log-scale fits and confidence bounds to WPM.
    y_linear = np.exp(y_linear)
    if plot_piecewise:
        y_spline, spline_lo, spline_hi = np.exp([y_spline, spline_lo, spline_hi])

    fig, ax = plt.subplots(figsize=(4, 4))

    sns.scatterplot(
        data=plot_df,
        x="episode",
        y="word_count_per_minute",
        hue="year",
        alpha=0.5,
        palette=palette,
        legend=None,
        s=30,
        ax=ax,
    )
    ax.plot(
        x_grid,
        y_linear,
        color="black",
        linewidth=2,
        linestyle="-",
        zorder=4,
        label="Linear Model",
    )
    if plot_piecewise:
        # Piecewise model: dashed line
        ax.plot(
            x_grid,
            y_spline,
            color="black",
            linewidth=2.2,
            linestyle="--",
            zorder=5,
            label="Piecewise Linear Model",
        )
        # 95% confidence band (only for the continuous piecewise line)
        ax.fill_between(
            x_grid, spline_lo, spline_hi, color="grey", alpha=0.25, linewidth=0, zorder=3
        )

    # Vertical lines for years
    year_positions = df.groupby("year")["episode"].min()
    for ep in year_positions:
        ax.axvline(x=ep, color="lightgrey", linestyle="-", alpha=0.5)
    if plot_piecewise:
        _add_knot(ax, knot)

    # Secondary x-axis for years
    secax = ax.secondary_xaxis("top")
    secax.set_xticks(year_positions.values)  # positions at first episode of each year
    secax.set_xticklabels(year_positions.index, fontsize=8)  # show year labels
    # secax.set_xlabel("Year")

    plt.xlim(min(df["episode"]) - 10, max(df["episode"]) + 10)
    plt.ylim(0, 160)
    plt.xlabel("Episodes")
    plt.ylabel("Word count per minute")
    ax.set_title("Narration density across episodes")
    _add_model_legend(ax)
    sns.despine()

    return fig, ax


def _add_model_legend(ax):
    """Top-left legend with only the model lines: light grey, transparent, rounded, no outline."""
    handles, labels = ax.get_legend_handles_labels()
    legend = ax.legend(
        handles,
        labels,
        loc="upper left",
        frameon=True,
        fancybox=True,
        framealpha=0.5,
        facecolor="lightgrey",
        edgecolor="none",
    )
    legend.get_frame().set_linewidth(0)
    return legend


def _has_piecewise_model(spline_model, knot):
    if (spline_model is None) != (knot is None):
        raise ValueError("A spline model and its knot must be provided together.")
    return spline_model is not None


def _add_knot(ax, knot):
    """Mark and label a knot inside the plot, independently of the y-axis limits."""
    ax.axvline(knot, color="grey", linestyle=":", linewidth=1, alpha=0.8)
    ax.text(
        knot,
        0.98,
        f"Ep. {knot:g}",
        rotation=90,
        va="top",
        ha="right",
        fontsize=6,
        color="dimgray",
        transform=ax.get_xaxis_transform(),
    )


def _fixed_effect_ci(model, design: np.ndarray, y_hat: np.ndarray, z: float = 1.96):
    """Pointwise CI of the fixed-effects prediction (design @ fe_params)."""
    k = len(model.fe_params)
    # Fixed-effect parameters come first in MixedLM's covariance matrix
    cov = np.asarray(model.cov_params())[:k, :k]
    se = np.sqrt(np.einsum("ij,jk,ik->i", design, cov, design))
    return y_hat - z * se, y_hat + z * se


def _fixed_params(model):
    """Return fixed effects for LMMs or regression parameters for GEE results."""
    return model.fe_params if hasattr(model, "fe_params") else model.params


def _episode_slope_name(parameter_names) -> str:
    for name in ("episode_scaled", "episode_shifted"):
        if name in parameter_names:
            return name
    raise ValueError("Model parameters must include an episode slope.")


def _model_prediction_ci(model, design: pd.DataFrame, z: float = 1.96):
    """Predict from fixed effects and return pointwise confidence bounds."""
    params = _fixed_params(model)
    matrix = design.loc[:, params.index].to_numpy()
    y_hat = matrix @ np.asarray(params)
    if hasattr(model, "fe_params"):
        lower, upper = _fixed_effect_ci(model, matrix, y_hat, z)
    else:
        covariance = np.asarray(model.cov_params())
        variance = np.einsum("ij,jk,ik->i", matrix, covariance, matrix)
        se = np.sqrt(np.maximum(variance, 0))
        lower, upper = y_hat - z * se, y_hat + z * se
    return y_hat, lower, upper


def plot_asl(
    df: pd.DataFrame,
    linear_model,
    spline_model,
    knot: int | None,
    episode_scale: float,
    aggregate_by_chapter: bool = False,
    y_max: float = 20,
    individual_shots: bool = False,
):
    """Create a shot-duration plot using fitted GEE or LMM model parameters.

    Args:
        df: Shot or chapter observations with 'episode', 'durations' and 'year'.
        linear_model: Fitted statsmodels GEE or MixedLM linear model.
        spline_model: Optional fitted statsmodels GEE or MixedLM spline model.
        knot: Best optimal turning point episode.
        aggregate_by_chapter: If True, average observations by 'chapter_uid' before plotting.
    """
    sort_cols = ["episode"] + (["chapter"] if "chapter" in df else [])
    plot_df = df.sort_values(sort_cols).reset_index(drop=True)
    episode_min = linear_model.model.data.frame["episode"].min()

    plot_piecewise = _has_piecewise_model(spline_model, knot)

    x_grid = np.linspace(
        plot_df["episode"].min(),
        plot_df["episode"].max(),
        500,
    )
    x_scaled = (x_grid - episode_min) / episode_scale

    linear_params = _fixed_params(linear_model)
    linear_slope_name = _episode_slope_name(linear_params.index)
    linear_design = pd.DataFrame(
        {
            "Intercept": np.ones_like(x_grid),
            linear_slope_name: x_scaled,
        }
    )
    y_linear, linear_lo, linear_hi = _model_prediction_ci(linear_model, linear_design)

    if plot_piecewise:
        spline_params = _fixed_params(spline_model)
        spline_slope_name = _episode_slope_name(spline_params.index)
        knot_scaled = (knot - episode_min) / episode_scale
        spline_1 = np.maximum(x_scaled - knot_scaled, 0)
        spline_design = pd.DataFrame(
            {
                "Intercept": np.ones_like(x_grid),
                spline_slope_name: x_scaled,
                "spline_1": spline_1,
            }
        )
        y_spline, _, _ = _model_prediction_ci(spline_model, spline_design)

    # 95% CI of the population-average line on the log scale.
    # Models were fitted on log(duration): back-transform to seconds
    # (exp of the fitted log-mean = geometric mean / median duration)
    y_linear, linear_lo, linear_hi = np.exp([y_linear, linear_lo, linear_hi])
    if plot_piecewise:
        y_spline = np.exp(y_spline)

    fig, ax = plt.subplots(figsize=(4, 4))

    if aggregate_by_chapter:
        scatter_df = plot_df.groupby(
            ["chapter_uid", "episode", "year"], as_index=False
        )["durations"].mean()
    else:
        scatter_df = plot_df
    point_alpha = 0.5
    point_size = 25

    sns.scatterplot(
        data=scatter_df,
        x="episode",
        y="durations",
        hue="year",
        alpha=point_alpha,
        palette=palette,
        legend=None,
        s=point_size,
        ax=ax,
    )
    # Linear
    ax.plot(
        x_grid,
        y_linear,
        color="black",
        linewidth=2,
        linestyle="-",
        zorder=4,
        label="Linear Model",
    )
    if plot_piecewise:
        ax.plot(
            x_grid,
            y_spline,
            color="black",
            linewidth=2.2,
            linestyle="--",
            zorder=5,
            label="Piecewise Linear Model",
        )
    # 95% confidence band (only for the continuous linear line)
    ax.fill_between(x_grid, linear_lo, linear_hi, color="grey", alpha=0.25, linewidth=0, zorder=3)

    # Vertical lines for years
    year_positions = df.groupby("year")["episode"].min()
    year_positions = year_positions[1:]
    for ep in year_positions:
        ax.axvline(x=ep, color="lightgrey", linestyle="-", alpha=0.5)

    if plot_piecewise:
        _add_knot(ax, knot)
    ax.set_ylim(bottom=0, top=y_max)

    # Outliers above the axis limit: triangles at the top edge, labelled with their value
    # outliers = scatter_df[scatter_df["durations"] > y_max]
    # if not outliers.empty:
    #     ax.scatter(
    #         outliers["episode"],
    #         np.full(len(outliers), y_max * 0.97),
    #         c=outliers["year"].map(palette),
    #         marker="^",
    #         s=point_size,
    #         edgecolors="black",
    #         linewidths=0.4,
    #         zorder=6,
    #     )
    #     for ep, dur in zip(outliers["episode"], outliers["durations"]):
    #         ax.annotate(
    #             f"{dur:.0f}",
    #             (ep, y_max * 0.97),
    #             xytext=(0, -4),
    #             textcoords="offset points",
    #             rotation=90,
    #             va="top",
    #             ha="center",
    #             fontsize=5,
    #             color="dimgray",
    #         )

    # Secondary x-axis for years
    secax = ax.secondary_xaxis("top")
    secax.set_xticks(year_positions.values)
    secax.set_xticklabels(year_positions.index, fontsize=8)

    plt.xlim(min(df["episode"]) - 10, max(df["episode"]) + 10)
    plt.xlabel("Episodes")
    plt.ylabel("Shot length (sec)" if individual_shots else "Avg. shot length (sec)")
    ax.set_title("Shot length across episodes")

    _add_model_legend(ax)
    sns.despine()

    return fig, ax


def _plot_stack_with_gee_model(
    data: pd.DataFrame,
    stack_cols: list[str],
    labels: list[str],
    curves: list[dict],
    episode_min: int,
    episode_scale: float,
    ylabel: str,
    title: str,
    window: int = 5,
):
    """Stackplot of smoothed per-episode class shares with fitted GEE curves on top.

    Args:
        data: Chapter shares or keyframe indicators with 'episode', 'year' and stack_cols.
            Values must sum to 1 per row. Episode averaging is for display only.
        curves: Dicts with
            'model': fitted GEE (episode_scaled and optionally spline_1),
            'boundary': 'bottom' (curve at P(class)) or 'top' (curve at 1 - P(class)),
            'kind': 'spline' or 'linear', 'label': legend label (optional),
            'color': line colour, 'knot': episode of the spline knot (spline only).
        episode_min, episode_scale: Scaling used to build 'episode_scaled' when fitting.
    """
    df_episode = data.groupby(["episode", "year"], as_index=False)[stack_cols].mean()
    df_episode = df_episode.sort_values("episode")
    df_plot = df_episode.copy()
    for col in stack_cols:
        df_plot[col] = df_plot[col].rolling(window=window, center=True, min_periods=1).mean()

    _, ax = plt.subplots(figsize=(6, 4))

    ax.stackplot(
        df_plot["episode"],
        *[df_plot[col] for col in stack_cols],
        labels=labels,
        colors=["#d7191c", "white", "#2b83ba"],
        alpha=0.6,
    )

    x_grid = np.linspace(data["episode"].min(), data["episode"].max(), 500)
    x_scaled = (x_grid - episode_min) / episode_scale
    drawn_knots = set()
    for curve in curves:
        params = curve["model"].params
        eta = params["Intercept"] + params["episode_scaled"] * x_scaled
        if curve["kind"] == "spline":
            knot_scaled = (curve["knot"] - episode_min) / episode_scale
            eta = eta + params["spline_1"] * np.maximum(x_scaled - knot_scaled, 0)
        elif curve["kind"] != "linear":
            raise ValueError(f"Unknown curve kind: {curve['kind']!r}")
        prob = 1 / (1 + np.exp(-eta))
        if curve["boundary"] == "top":
            prob = 1 - prob
        ax.plot(
            x_grid,
            prob,
            color=curve["color"],
            linewidth=2,
            linestyle="--" if curve["kind"] == "spline" else "-",
            zorder=5,
            label=curve.get("label"),
        )
        if curve["kind"] == "spline" and curve["knot"] not in drawn_knots:
            drawn_knots.add(curve["knot"])
            _add_knot(ax, curve["knot"])

    # Vertical lines for years
    year_positions = data.groupby("year")["episode"].min()
    for ep in year_positions:
        ax.axvline(x=ep, color="lightgrey", linestyle="-", alpha=0.5)

    secax = ax.secondary_xaxis("top")
    secax.set_xticks(year_positions.values)
    secax.set_xticklabels(year_positions.index)
    secax.set_xlabel("Year")

    ax.set_xlabel("Episode")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_ylim(0, 1)
    ax.set_xlim(data["episode"].min(), data["episode"].max())

    leg = ax.legend(loc="upper right", frameon=True, fontsize=9)
    leg.get_frame().set_facecolor("white")
    leg.get_frame().set_alpha(0.7)

    return plt


def plot_shot_scale_per_episode(
    data: pd.DataFrame,
    model_close_linear,
    model_close_spline,
    model_long_linear,
    model_long_spline,
    episode_min: int,
    episode_scale: float,
):
    """Shot scale shares with linear and automated-knot GEE models.

    Args:
        data: Chapter shares or keyframe indicators with 'episode', 'year', 'is_close', 'is_long'.
        Each spline model must have its selected episode in 'knot_found'.
    """
    df = data.copy()
    df["is_middle"] = 1 - df["is_close"] - df["is_long"]
    return _plot_stack_with_gee_model(
        df,
        stack_cols=["is_close", "is_middle", "is_long"],
        labels=["Close-up", "Medium Shot", "Long Shot"],
        curves=[
            {
                "model": model_close_linear,
                "boundary": "bottom",
                "kind": "linear",
                "color": RED,
                "label": "Close-up: Linear Model",
            },
            # {
            #     "model": model_close_spline,
            #     "boundary": "bottom",
            #     "kind": "spline",
            #     "color": RED,
            #     "label": "Close-up: Piecewise Model",
            #     "knot": model_close_spline.knot_found,
            # },
            {
                "model": model_long_linear,
                "boundary": "bottom",
                "kind": "linear",
                "color": BLUE,
                "label": "Long Shot: Linear Model",
            },
            # {
            #     "model": model_long_spline,
            #     "boundary": "bottom",
            #     "kind": "spline",
            #     "color": BLUE,
            #     "label": "Long Shot: Piecewise Model",
            #     "knot": model_long_spline.knot_found,
            # },
        ],
        episode_min=episode_min,
        episode_scale=episode_scale,
        ylabel="Shares of Shot Scales",
        title="Development of Shot Scale Shares over time",
    )


def plot_depiction_ratio_per_episode(
    data: pd.DataFrame,
    model_individual_linear,
    model_individual_spline,
    model_collective_linear,
    model_collective_spline,
    episode_min: int,
    episode_scale: float,
):
    """Soldier depiction shares per episode with GEE trends.

    Both categories show a solid linear and a dashed automated-knot model.

    Args:
        data: Chapter shares or keyframe indicators with 'episode', 'year' and class columns.
        Each spline model must have its selected episode in 'knot_found'.
    """
    df = data.copy()
    df["is_multiple"] = 1 - df["is_individual"] - df["is_collective"]
    return _plot_stack_with_gee_model(
        df,
        stack_cols=["is_individual", "is_multiple", "is_collective"],
        labels=["Individual", "Multiple", "Collective"],
        curves=[
            {
                "model": model_individual_linear,
                "boundary": "bottom",
                "kind": "linear",
                "color": RED,
                "label": "Individual: Linear Model",
            },
            # {
            #     "model": model_individual_spline,
            #     "boundary": "bottom",
            #     "kind": "spline",
            #     "color": RED,
            #     "label": "Individual: Piecewise Model",
            #     "knot": model_individual_spline.knot_found,
            # },
            {
                "model": model_collective_linear,
                "boundary": "bottom",
                "kind": "linear",
                "color": BLUE,
                "label": "Collective: Linear Model",
            },
            {
                "model": model_collective_spline,
                "boundary": "bottom",
                "kind": "spline",
                "color": BLUE,
                "label": "Collective: Piecewise Model",
                "knot": model_collective_spline.knot_found,
            },
        ],
        episode_min=episode_min,
        episode_scale=episode_scale,
        ylabel="Shares of Soldier Depictions",
        title="Development of Soldier Depiction Shares over time",
    )




def plot_residual_qq(
    residuals: np.ndarray,
    title: str,
    ylabel: str = "Conditional residual quantiles (model scale)",
):
    fig, ax = plt.subplots(figsize=(4, 4))
    qqplot(residuals, line="s", ax=ax)
    ax.set_xlabel("Theoretical normal quantiles")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    return fig, ax
