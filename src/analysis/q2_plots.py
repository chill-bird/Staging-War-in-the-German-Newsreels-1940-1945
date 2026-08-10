"""
q2_plots.py
---

Plots for q2_formal_features
"""

from pathlib import Path

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from PIL import Image
from tueplots import bundles

from src.util import get_keyframe_path

plt.rcParams.update(bundles.beamer_moml())
plt.rcParams.update(
    {
        "figure.dpi": 200,
        # "figure.figsize": (5.5324802778 / 1.5, 8.6938975 / 3*1.5),
        "font.size": 10,  # base font size
        "axes.titlesize": 14,  # title size
        "axes.labelsize": 12,  # x and y label size
        "xtick.labelsize": 10,  # x tick labels
        "ytick.labelsize": 10,  # y tick labels
        "legend.fontsize": 10,  # legend text
    }
)
# Color palette
years = [1940, 1941, 1942, 1943, 1944, 1945]
palette = dict(zip(years, sns.color_palette("tab20", len(years))))
palette[1945] = "gold"  # Override 1945


def plot_depiction_ratio_per_episode(
    df: pd.DataFrame, spearman_p: float, spearman_r: float, median: float, mean: float, std: float
):
    """Creates plot for soldier ratio per episode"""

    description = "Development of Soldier Depiction Shares over time"

    # Aggregate per episode
    df_episode = (
        df.groupby(["episode", "year"], as_index=False)
        .agg(
            {
                "ratio_individual_depictions": "mean",
                "ratio_group_depictions": "mean",
                "ratio_collective_depictions": "mean",
            }
        )
        .sort_values("episode")
    )

    # Smooth with rolling mean
    window = 5
    for col in [
        "ratio_individual_depictions",
        "ratio_group_depictions",
        "ratio_collective_depictions",
    ]:
        df_episode[col] = df_episode[col].rolling(window=window, center=True, min_periods=1).mean()

    # Figure and axes
    fig, ax = plt.subplots(figsize=(6, 4))

    text = (
        f"Spearman ρ = {spearman_r:.2f} (p = {spearman_p:.3g})\n"
        f"Mean: {mean:.2f}\n"
        f"Standard deviation: {std:.2f}"
    )

    # Stackplot
    ax.stackplot(
        df_episode["episode"],
        df_episode["ratio_individual_depictions"],
        df_episode["ratio_group_depictions"],
        df_episode["ratio_collective_depictions"],
        labels=["Individual", "Multiple", "Collective"],
        colors=["#d7191c", "white", "#2b83ba"],
        alpha=0.6,
    )

    # Regression lines
    sns.regplot(
        data=df,
        x="episode",
        y="ratio_individual_depictions",
        scatter=False,
        label="Individual Depictions Trend",
        color="#d7191c",
        ax=ax,
    )
    sns.regplot(
        data=df,
        x="episode",
        y="ratio_group_depictions",
        scatter=False,
        label="Multiple Depictions Trend",
        color="lightgrey",
        ax=ax,
    )
    sns.regplot(
        data=df,
        x="episode",
        y="ratio_collective_depictions",
        scatter=False,
        label="Collective Depictions Trend",
        color="#2b83ba",
        ax=ax,
    )

    # Vertical lines for years
    year_positions = df.groupby("year")["episode"].min()
    for ep in year_positions:
        ax.axvline(x=ep, color="lightgrey", linestyle="-", alpha=0.5)

    # Secondary x-axis for years
    secax = ax.secondary_xaxis("top")
    secax.set_xticks(year_positions.values)  # positions at first episode of each year
    secax.set_xticklabels(year_positions.index)  # show year labels
    secax.set_xlabel("Year")

    ax.text(
        0.05,
        0.95,
        text,
        transform=plt.gca().transAxes,
        verticalalignment="top",
        bbox=dict(boxstyle="round", color="lightgrey", alpha=0.4),
    )
    ax.set_xlabel("Episode")
    ax.set_ylabel("Shares of Soldier Depictions")
    ax.set_title(f"{description} (n={len(df_episode)})")
    ax.set_ylim(0, 1)
    ax.set_xlim(df["episode"].min(), df["episode"].max())

    leg = ax.legend(loc="upper right", frameon=True, fontsize=9)
    leg.get_frame().set_facecolor("white")
    leg.get_frame().set_alpha(0.7)

    return plt


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
