"""
q1_plots.py
---

Plots for q1_formal_features
"""

from pathlib import Path

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from PIL import Image
from tueplots import bundles

# from tueplots import figsizes
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


def plot_word_counts_per_episode(
    df: pd.DataFrame, spearman_p: float, spearman_r: float, median: float, mean: float, std: float
):
    """Creates plot for word counts per episode"""

    plot_title = "Rel. Word Counts in DWS chapters"

    df_sorted = df.sort_values(["episode", "chapter"]).reset_index(drop=True)

    text = (
        f"Spearman ρ = {spearman_r:.2f} (p = {spearman_p:.3g})\n"
        f"Median: {median:.2f}\n"
        f"Mean: {mean:.2f}\n"
        f"Standard deviation: {std:.2f}"
    )

    fig, ax = plt.subplots(figsize=(4, 4))

    sns.scatterplot(
        data=df_sorted,
        x="episode",
        y="word_count_per_minute",
        hue="year",
        alpha=0.5,
        palette=palette,
        legend=None,
        s=30,
    )

    sns.regplot(
        data=df_sorted,
        x="episode",
        y="word_count_per_minute",
        scatter=False,
        order=1,
        line_kws={"linewidth": 2, "color": "black"},
        ax=ax,
    )

    plt.text(
        0.05,
        0.95,
        text,
        transform=plt.gca().transAxes,
        verticalalignment="top",
        bbox=dict(boxstyle="round", color="lightgrey", alpha=0.4),
    )

    # Vertical lines for years
    year_positions = df.groupby("year")["episode"].min()
    year_positions = year_positions[1:]
    for ep in year_positions:
        ax.axvline(x=ep, color="lightgrey", linestyle="-", alpha=0.5)

    # Secondary x-axis for years
    secax = ax.secondary_xaxis("top")
    secax.set_xticks(year_positions.values)  # positions at first episode of each year
    secax.set_xticklabels(year_positions.index, fontsize=8)  # show year labels
    # secax.set_xlabel("Year")

    plt.xlim(min(df["episode"]) - 10, max(df["episode"]) + 10)
    plt.ylim(0, 160)
    plt.xlabel("Episodes")
    plt.ylabel("Word Count per minute")
    plt.title(f"{plot_title} (n={len(df_sorted)})")
    # sns.move_legend(
    #     ax,
    #     "lower center",
    #     bbox_to_anchor=(0.5, -0.2),
    #     ncol=6,
    #     title=None,
    #     frameon=False,
    #     # sns.move_legend(ax, "upper right", bbox_to_anchor=(1, 1))
    # )
    return plt


def plot_shot_duration_per_episode(
    df: pd.DataFrame, spearman_p: float, spearman_r: float, median: float, mean: float, std: float
):
    """Creates plot for shot duration per episode"""

    plot_title = "Shot Lengths in DWS chapters"

    df_sorted = df.sort_values(["episode", "chapter"]).reset_index(drop=True)

    text = (
        f"Spearman ρ = {spearman_r:.2f} (p = {spearman_p:.3g})\n"
        f"Median: {median:.2f}\n"
        f"Mean: {mean:.2f}\n"
        f"Standard deviation: {std:.2f}"
    )

    fig, ax = plt.subplots(figsize=(4, 4))

    ax = sns.scatterplot(
        data=df_sorted,
        x="episode",
        y="avg_shot_duration",
        hue="year",
        alpha=0.5,
        palette=palette,
        legend=None,
        s=30,
    )

    sns.regplot(
        data=df_sorted,
        x="episode",
        y="avg_shot_duration",
        scatter=False,
        order=1,
        line_kws={"linewidth": 2, "color": "black"},
        ax=ax,
    )

    plt.text(
        0.05,
        0.95,
        text,
        transform=plt.gca().transAxes,
        verticalalignment="top",
        bbox=dict(boxstyle="round", color="lightgrey", alpha=0.4),
    )

    # Vertical lines for years
    year_positions = df.groupby("year")["episode"].min()
    year_positions = year_positions[1:]
    for ep in year_positions:
        ax.axvline(x=ep, color="lightgrey", linestyle="-", alpha=0.5)

    # Secondary x-axis for years
    secax = ax.secondary_xaxis("top")
    secax.set_xticks(year_positions.values)  # positions at first episode of each year
    secax.set_xticklabels(year_positions.index, fontsize=8)  # show year labels
    # secax.set_xlabel("Year")

    plt.xlim(min(df["episode"]) - 10, max(df["episode"]) + 10)
    plt.ylim(0, 17)
    plt.xlabel("Episodes")
    plt.ylabel("Average Shot Length in sec")
    plt.title(f"{plot_title} (n={len(df_sorted)})")
    # sns.move_legend(
    #     ax,
    #     "lower center",
    #     bbox_to_anchor=(0.5, -0.2),
    #     ncol=6,
    #     title=None,
    #     frameon=False,
    #     # sns.move_legend(ax, "upper right", bbox_to_anchor=(1, 1))
    # )

    return plt


def plot_shot_scale_per_episode(
    df: pd.DataFrame, spearman_p: float, spearman_r: float, median: float, mean: float, std: float
):
    """Creates plot for shot scale ratio per episode"""

    plot_title = "Development of Shot Scale Shares over time"

    # Aggregate per episode
    df_episode = (
        df.groupby(["episode", "year"], as_index=False)
        .agg({"ratio_close_shots": "mean", "ratio_long_shots": "mean"})
        .sort_values("episode")
    )
    df_episode["ratio_middle"] = (
        1 - df_episode["ratio_close_shots"] - df_episode["ratio_long_shots"]
    )

    # Smooth with rolling mean
    window = 5
    for col in ["ratio_close_shots", "ratio_long_shots", "ratio_middle"]:
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
        df_episode["ratio_close_shots"],
        df_episode["ratio_middle"],
        df_episode["ratio_long_shots"],
        labels=["Close Shot", "Medium Shot", "Long Shot"],
        colors=["#d7191c", "white", "#2b83ba"],
        alpha=0.6,
    )

    # Regression lines
    sns.regplot(
        data=df,
        x="episode",
        y="ratio_close_shots",
        scatter=False,
        label="Close Shot trend",
        color="#d7191c",
        ax=ax,
    )
    sns.regplot(
        data=df,
        x="episode",
        y="ratio_long_shots",
        scatter=False,
        label="Long Shot Trend",
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
    ax.set_ylabel("Shares of Shot Scales")
    ax.set_title(f"{plot_title} (n={len(df_episode)})")
    ax.set_ylim(0, 1)
    ax.set_xlim(df["episode"].min(), df["episode"].max())

    leg = ax.legend(loc="upper right", frameon=True, fontsize=9)
    leg.get_frame().set_facecolor("white")
    leg.get_frame().set_alpha(0.7)

    return plt


def plot_top_to_bottom_similarities(df: pd.DataFrame):
    """Creates plot for top, medium and bottom classification of keyframes regarding shot scale,
    with varying row heights and preserved image aspect ratios."""

    def get_row_heights(df, shot_scales, max_width=400):
        """Compute relative heights for each row based on tallest image in row."""
        heights = []
        for shot_scale, _ in shot_scales:
            subset = df[df["shot_scale_class"] == shot_scale].sort_values(
                "shot_scale_sim_score", ascending=False
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

    shot_scales = [
        ("Long Shot", "WS"),
        ("Medium Shot", "MS"),
        ("Close-up", "CS"),
    ]

    # relative heights for the rows (can tweak as needed)
    row_heights = get_row_heights(df, shot_scales)

    fig = plt.figure(figsize=(4, 4), constrained_layout=False)

    # outer grid for rows with varying heights
    outer = gridspec.GridSpec(3, 1, figure=fig, height_ratios=row_heights, hspace=0.00, wspace=0.0)

    for row_idx, (shot_scale, label) in enumerate(shot_scales):
        subset = df[df["shot_scale_class"] == shot_scale].sort_values(
            "shot_scale_sim_score", ascending=False
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

            sim_score = float(row["shot_scale_sim_score"])
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
