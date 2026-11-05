"""
lmm_analysis_plot.py
---

Plots for lmm_analysis
"""

from pathlib import Path

from matplotlib.lines import Line2D
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from matplotlib import font_manager
from PIL import Image
from tueplots import bundles

# from tueplots import figsizes
from src.util import get_keyframe_path

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


def plot_word_counts_per_episode_automated_best_knot(
    df: pd.DataFrame, quadratic_model, spline_model, knot: int
):
    """Creates plot for word counts per episode"""

    plot_df = df.sort_values(["episode", "chapter"]).reset_index(drop=True)
    model_df = spline_model.model.data.frame
    episode_origin = float((model_df["episode"] - model_df["episode_shifted"]).iloc[0])

    x_grid = np.linspace(
        plot_df["episode"].min(),
        plot_df["episode"].max(),
        500,
    )
    x_centered = x_grid - episode_origin

    quadratic_fe = quadratic_model.fe_params
    quadratic_intercept = quadratic_fe["Intercept"]
    quadratic_slope = quadratic_fe["episode_shifted"]
    quadratic_term = quadratic_fe["I(episode_shifted ** 2)"]
    y_quadratic = (
        quadratic_intercept
        + quadratic_slope * x_centered
        + quadratic_term * x_centered**2
    )

    spline_fe = spline_model.fe_params
    spline_intercept = spline_fe["Intercept"]
    spline_slope = spline_fe["episode_shifted"]
    spline_change = spline_fe["spline_1"]
    # Post-breakpoint slope
    spline_post_slope = spline_slope + spline_change
    spline_1 = np.maximum(x_grid - knot, 0)
    y_spline = spline_intercept + spline_slope * x_centered + spline_change * spline_1
    spline_lo, spline_hi = _fixed_effect_ci(
        spline_model,
        np.column_stack([np.ones_like(x_grid), x_centered, spline_1]),
        y_spline,
    )

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
    # Quadratic
    ax.plot(
        x_grid,
        y_quadratic,
        color="black",
        linewidth=2,
        linestyle="-",
        zorder=4,
        label="Quadratic Model",
    )
    # Piecewise mixed-effects model: solid line
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
    ax.axvline(
        knot,
        color="grey",
        linestyle=":",
        linewidth=1,
        alpha=0.8,
    )
    # Axes-fraction y keeps the label inside the axes regardless of data limits
    ax.text(
        knot,
        0.98,
        f"Ep. {knot}",
        rotation=90,
        va="top",
        ha="right",
        fontsize=6,
        color="dimgray",
        transform=ax.get_xaxis_transform(),
    )

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


def _fixed_effect_ci(model, design: np.ndarray, y_hat: np.ndarray, z: float = 1.96):
    """Pointwise CI of the fixed-effects prediction (design @ fe_params)."""
    k = len(model.fe_params)
    # Fixed-effect parameters come first in MixedLM's covariance matrix
    cov = np.asarray(model.cov_params())[:k, :k]
    se = np.sqrt(np.einsum("ij,jk,ik->i", design, cov, design))
    return y_hat - z * se, y_hat + z * se


def plot_shot_durations_per_episode_automated_best_knot(
    df: pd.DataFrame,
    linear_model,
    spline_model,
    knot: int,
    aggregate_by_chapter: bool = False,
    y_max: float = 20,
):
    """Creates plot for shot durations per episode using LMM spline model.

    Args:
        df: Exploded shot durations DataFrame containing 'episode', 'durations', 'year', 'chapter'.
        spline_model: Fitted statsmodels MixedLM object with fixed effects.
        knot: Best optimal turning point episode.
        aggregate_by_chapter: If True, plot chapter mean durations; if False, plot all individual shot points.
    """
    plot_df = df.sort_values(["episode", "chapter"]).reset_index(drop=True)
    model_df = spline_model.model.data.frame
    episode_origin = float((model_df["episode"] - model_df["episode_shifted"]).iloc[0])

    x_grid = np.linspace(
        plot_df["episode"].min(),
        plot_df["episode"].max(),
        500,
    )
    x_centered = x_grid - episode_origin

    linear_fe = linear_model.fe_params
    linear_intercept = linear_fe["Intercept"]
    linear_slope = linear_fe["episode_shifted"]
    y_linear = linear_intercept + linear_slope * x_centered

    spline_fe = spline_model.fe_params
    spline_intercept = spline_fe["Intercept"]
    spline_slope = spline_fe["episode_shifted"]
    spline_change = spline_fe["spline_1"]
    # Post-breakpoint slope
    spline_post_slope = spline_slope + spline_change
    spline_1 = np.maximum(x_grid - knot, 0)
    y_spline = spline_intercept + spline_slope * x_centered + spline_change * spline_1

    # 95% CI of the fixed-effects line on the log scale
    ones = np.ones_like(x_grid)
    linear_lo, linear_hi = _fixed_effect_ci(
        linear_model, np.column_stack([ones, x_centered]), y_linear
    )

    # Models were fitted on log(duration): back-transform to seconds
    # (exp of the fitted log-mean = geometric mean / median duration)
    y_linear, linear_lo, linear_hi = np.exp([y_linear, linear_lo, linear_hi])
    y_spline = np.exp(y_spline)

    fig, ax = plt.subplots(figsize=(4, 4))

    # Prepare scatter data
    if aggregate_by_chapter:
        scatter_df = plot_df.groupby(["episode", "chapter", "year"], as_index=False)[
            "durations"
        ].mean()
        point_alpha = 0.5
        point_size = 25
    else:
        scatter_df = plot_df
        point_alpha = 0.2  # Higher transparency for dense individual shot points
        point_size = 12

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
        label="Linear Mixed Model",
    )
    # Piecewise mixed-effects model: dashed line
    # ax.plot(
    #     x_grid,
    #     y_spline,
    #     color="black",
    #     linewidth=2.2,
    #     linestyle="--",
    #     zorder=5,
    #     label="Piecewise Linear Mixed Model",
    # )
    # 95% confidence band (only for the continuous linear line)
    ax.fill_between(
        x_grid, linear_lo, linear_hi, color="grey", alpha=0.25, linewidth=0, zorder=3
    )

    # Vertical lines for years
    year_positions = df.groupby("year")["episode"].min()
    year_positions = year_positions[1:]
    for ep in year_positions:
        ax.axvline(x=ep, color="lightgrey", linestyle="-", alpha=0.5)

    ax.axvline(
        knot,
        color="grey",
        linestyle=":",
        linewidth=1,
        alpha=0.8,
    )
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

    # Axes-fraction y keeps the label inside the axes regardless of data limits
    ax.text(
        knot,
        0.98,
        f"Ep. {knot}",
        rotation=90,
        va="top",
        ha="right",
        fontsize=6,
        color="dimgray",
        transform=ax.get_xaxis_transform(),
    )

    # Secondary x-axis for years
    secax = ax.secondary_xaxis("top")
    secax.set_xticks(year_positions.values)
    secax.set_xticklabels(year_positions.index, fontsize=8)

    plt.xlim(min(df["episode"]) - 10, max(df["episode"]) + 10)
    plt.xlabel("Episodes")
    plt.ylabel("Avg. shot length (sec)")
    ax.set_title("Shot length across episodes")

    _add_model_legend(ax)
    sns.despine()

    return fig, ax
