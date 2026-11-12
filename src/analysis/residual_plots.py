"""Shared residual Q-Q plots for the analysis models."""

import matplotlib.pyplot as plt
import numpy as np
from statsmodels.graphics.gofplots import qqplot

from src.util import DOC_DIR

RESIDUALS_DIR = DOC_DIR / "imgs" / "residuals"


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
