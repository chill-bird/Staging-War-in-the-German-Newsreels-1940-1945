"""
q2_plots.py
---

Plots for q2_formal_features
"""

from pathlib import Path

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import font_manager
from PIL import Image
from tueplots import bundles

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


RED = "#d7191c"
BLUE = "#2b83ba"


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
    """Stackplot of per-episode class shares with fitted GEE curves on top.

    Args:
        data: Keyframe-level DataFrame with 'episode', 'year' and the 0/1 columns in stack_cols
            (listed from the bottom to the top of the stack; they must sum to 1 per row).
        curves: Dicts with
            'model': fitted GEE (episode_scaled [+ spline_1 | I(episode_scaled ** 2)]),
            'boundary': 'bottom' (curve at P(class)) or 'top' (curve at 1 - P(class)),
            'kind': 'spline' or 'quadratic',
            'linestyle': matplotlib linestyle, 'label': legend label (optional),
            'color': line colour, 'knot': (episode, label) for a vertical knot line (optional, spline only).
        episode_min, episode_scale: Scaling used to build 'episode_scaled' when fitting.
    """
    df_episode = data.groupby(["episode", "year"], as_index=False)[stack_cols].mean()
    df_episode = df_episode.sort_values("episode")
    df_plot = df_episode.copy()
    for col in stack_cols:
        df_plot[col] = df_plot[col].rolling(window=window, center=True, min_periods=1).mean()

    fig, ax = plt.subplots(figsize=(6, 4))

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
            knot_scaled = (curve["knot"][0] - episode_min) / episode_scale
            eta = eta + params["spline_1"] * np.maximum(x_scaled - knot_scaled, 0)
        else:
            quad_name = next(name for name in params.index if "**2" in name.replace(" ", ""))
            eta = eta + params[quad_name] * x_scaled**2
        prob = 1 / (1 + np.exp(-eta))
        if curve["boundary"] == "top":
            prob = 1 - prob
        ax.plot(
            x_grid,
            prob,
            color=curve["color"],
            linewidth=2,
            linestyle=curve["linestyle"],
            zorder=5,
            label=curve.get("label"),
        )
        if curve["kind"] == "spline" and curve["knot"] not in drawn_knots:
            drawn_knots.add(curve["knot"])
            knot_episode, knot_label = curve["knot"]
            ax.axvline(knot_episode, color="grey", linestyle=":", linewidth=1, alpha=0.8)
            # Axes-fraction y keeps the label inside the axes regardless of data limits
            ax.text(
                knot_episode,
                0.98,
                knot_label,
                rotation=90,
                va="top",
                ha="right",
                fontsize=6,
                color="dimgray",
                transform=ax.get_xaxis_transform(),
            )

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
    model_close,
    model_long,
    barbarossa_episode: int,
    episode_min: int,
    episode_scale: float,
):
    """Shot scale shares per episode with Barbarossa-knot GEE models for close-ups and long shots.

    Args:
        data: Keyframe-level data with 'episode', 'year', 'is_close', 'is_long'.
        model_close, model_long: One-knot GEE models fitted on 'is_close' / 'is_long'.
    """
    df = data.copy()
    df["is_middle"] = 1 - df["is_close"] - df["is_long"]
    knot = (barbarossa_episode, "Barbarossa")
    return _plot_stack_with_gee_model(
        df,
        stack_cols=["is_close", "is_middle", "is_long"],
        labels=["Close-up", "Medium Shot", "Long Shot"],
        curves=[
            {
                "model": model_close,
                "boundary": "bottom",
                "kind": "spline",
                "linestyle": "-",
                "color": RED,
                "label": "Close-up model",
                "knot": knot,
            },
            {
                "model": model_long,
                "boundary": "bottom",
                "kind": "spline",
                "linestyle": "-",
                "color": BLUE,
                "label": "Long Shot model",
                "knot": knot,
            },
        ],
        episode_min=episode_min,
        episode_scale=episode_scale,
        ylabel="Shares of Shot Scales",
        title="Development of Shot Scale Shares over time",
    )


def plot_depiction_ratio_per_episode(
    data: pd.DataFrame,
    model_individual_spline,
    model_individual_quadratic,
    model_collective,
    barbarossa_episode: int,
    episode_min: int,
    episode_scale: float,
):
    """Soldier depiction shares per episode with GEE trends.

    Individuals: automated best-knot piecewise model (solid) and quadratic model (dashed).
    Collectives: Barbarossa-knot piecewise model (solid).

    Args:
        data: Keyframe-level data with 'episode', 'year', 'is_individual', 'is_collective'.
        model_individual_spline: Best-knot GEE on 'is_individual' (has 'knot_found').
        model_individual_quadratic: Quadratic GEE on 'is_individual'.
        model_collective: Barbarossa-knot GEE on 'is_collective'.
    """
    df = data.copy()
    df["is_multiple"] = 1 - df["is_individual"] - df["is_collective"]
    best_knot = int(model_individual_spline.knot_found)
    return _plot_stack_with_gee_model(
        df,
        stack_cols=["is_individual", "is_multiple", "is_collective"],
        labels=["Individual", "Multiple", "Collective"],
        curves=[
            {
                "model": model_individual_spline,
                "boundary": "bottom",
                "kind": "spline",
                "linestyle": "-",
                "color": RED,
                "label": "Individual: Piecewise Model",
                "knot": (best_knot, f"Ep. {best_knot}"),
            },
            {
                "model": model_individual_quadratic,
                "boundary": "bottom",
                "kind": "quadratic",
                "linestyle": "--",
                "color": RED,
                "label": "Individual: Quadratic Model",
            },
            {
                "model": model_collective,
                "boundary": "bottom",
                "kind": "spline",
                "linestyle": "-",
                "color": BLUE,
                "label": "Collective: Piecewise Model",
                "knot": (barbarossa_episode, "Barbarossa"),
            },
        ],
        episode_min=episode_min,
        episode_scale=episode_scale,
        ylabel="Shares of Soldier Depictions",
        title="Development of Soldier Depiction Shares over time",
    )


def plot_top_to_bottom_similarities(df: pd.DataFrame):
    """Creates plot for top, medium and bottom classification of keyframes regarding shot scale,
    with varying row heights and preserved image aspect ratios."""

    def get_row_heights(df, depictions, max_width=400):
        """Compute relative heights for each row based on tallest image in row."""
        heights = []
        for shot_scale, _ in depictions:
            subset = df[df["content_class"] == shot_scale].sort_values(
                "content_sim_score", ascending=False
            )
            if subset.empty:
                heights.append(1)
                continue

            rows = [
                subset.iloc[0],
                subset.iloc[len(subset) // 2],
                subset.iloc[-1],
            ]

            # find max height-to-width ratio in this row
            ratios = []
            for row in rows:
                try:
                    img_path = Path(get_keyframe_path(row["filestem"], row["frame"])).resolve()
                    img = Image.open(img_path)
                    ratios.append(img.height / img.width)
                except Exception:
                    ratios.append(1)  # fallback if image missing
            # use max ratio as row height proxy
            heights.append(max(ratios))
        return heights

    plot_title = "Top, Medium, Bottom Sim Scores"

    depictions = [
        ("collective", "Koll"),
        ("multiples", "Mult"),
        ("individual", "Indv"),
    ]

    # relative heights for the rows (can tweak as needed)
    row_heights = get_row_heights(df, depictions)

    fig = plt.figure(figsize=(4, 4), constrained_layout=False)

    # outer grid for rows with varying heights
    outer = gridspec.GridSpec(3, 1, figure=fig, height_ratios=row_heights, hspace=0.00, wspace=0.0)

    for row_idx, (depiction, label) in enumerate(depictions):
        subset = df[df["content_class"] == depiction].sort_values(
            "content_sim_score", ascending=False
        )

        if subset.empty:
            continue

        rows = [
            subset.iloc[0],
            subset.iloc[len(subset) // 2],
            subset.iloc[-1],
        ]

        # inner grid: 1 row, 4 columns (label + 3 images)
        inner = gridspec.GridSpecFromSubplotSpec(
            1, 4, subplot_spec=outer[row_idx], width_ratios=[0.10, 1, 1, 1], wspace=0.03, hspace=0.0
        )

        # ---- left label axis ----
        label_ax = fig.add_subplot(inner[0])
        label_ax.text(
            0.5,
            0.5,
            label,
            rotation=90,
            ha="center",
            va="center",
            fontsize=12,
            fontweight="bold",
        )
        label_ax.axis("off")

        # ---- image axes ----
        for col_idx, row in enumerate(rows):
            ax = fig.add_subplot(inner[col_idx + 1])
            ax.set_anchor("C")

            sim_score = float(row["content_sim_score"])
            img_path = Path(get_keyframe_path(row["filestem"], row["frame"])).resolve()

            try:
                img = Image.open(img_path)
                ax.imshow(img, aspect="auto")  # aspect auto for scaling within ax

                # preserve aspect ratio by adjusting extent of the image
                ax.set_aspect("equal")

            except Exception:
                ax.text(
                    0.5,
                    0.5,
                    "Image not found",
                    ha="center",
                    va="center",
                    transform=ax.transAxes,
                )

            ax.text(
                0.2,
                0.3,
                f"{sim_score:.2f}",
                transform=ax.transAxes,
                ha="center",
                va="center",
                fontsize=12,
                fontweight="bold",
                color="gold",
                bbox=dict(
                    facecolor="lightgrey", alpha=0.5, boxstyle="round,pad=0.2", edgecolor="none"
                ),
            )

            ax.axis("off")

    fig.subplots_adjust(left=0.02, right=0.98, top=0.92, bottom=0.02, hspace=0.0, wspace=0.0)

    fig.suptitle(plot_title, fontsize=14)
    plt.rcParams["axes.xmargin"] = 0
    plt.rcParams["axes.ymargin"] = 0

    return plt
