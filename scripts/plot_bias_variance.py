"""Slide figure for the XGBoost model choice.

Each of the 72 grid settings is one point. The horizontal axis is the
average of the train, Test 1, and Test 2 AUCs. Farther right means less
bias. The vertical axis is the spread of those three AUCs. Higher means
more variance.

The selection score is average minus spread. The dashed line is the
winning score. The chosen model sits on that line, and every other model
is above it.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parent.parent
GRID_PATH = PROJECT_DIR / "results" / "05_xgb_grid_bias_variance.csv"
OUTPUT_PATH = PROJECT_DIR / "results" / "08_xgb_bias_variance.png"

# Large enough to read when the image is dropped onto a slide.
plt.rcParams.update({
    "font.size": 16,
    "axes.titlesize": 20,
    "axes.labelsize": 18,
    "xtick.labelsize": 14,
    "ytick.labelsize": 14,
    "legend.fontsize": 14,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.facecolor": "white",
    "axes.edgecolor": "#222222",
    "text.color": "#222222",
    "axes.labelcolor": "#222222",
    "xtick.color": "#222222",
    "ytick.color": "#222222",
})


def main():
    grid = pd.read_csv(GRID_PATH)
    chosen = grid.loc[grid["chosen"]].iloc[0]
    others = grid.loc[~grid["chosen"]]
    score = float(chosen["selection_score"])

    figure, axis = plt.subplots(figsize=(11, 7.2))
    axis.scatter(
        others["mean_auc"],
        others["std_auc"],
        s=70,
        color="#8A8F98",
        label="Other settings",
        zorder=2,
    )
    axis.scatter(
        [chosen["mean_auc"]],
        [chosen["std_auc"]],
        s=160,
        color="#C2410C",
        label="Chosen model",
        zorder=3,
    )

    # Points on this line have the same score as the winner.
    # score = average - spread, so spread = average - score.
    x_left = float(grid["mean_auc"].min()) - 0.001
    x_right = float(grid["mean_auc"].max()) + 0.001
    axis.plot(
        [x_left, x_right],
        [x_left - score, x_right - score],
        linestyle="--",
        color="#222222",
        linewidth=1.2,
        label="Winning score",
        zorder=1,
    )

    axis.set_xlabel("Average AUC")
    axis.set_ylabel("Spread of AUC")
    axis.set_title("Chosen model: highest average minus spread")
    axis.set_xlim(x_left, x_right)
    axis.set_ylim(0, float(grid["std_auc"].max()) + 0.002)
    axis.xaxis.set_major_formatter(plt.FuncFormatter(lambda value, _position: f"{value:.3f}"))
    axis.yaxis.set_major_formatter(plt.FuncFormatter(lambda value, _position: f"{value:.3f}"))
    axis.legend(frameon=False, loc="upper left")
    axis.grid(True, color="#E6E6E6", linewidth=0.8)
    axis.set_axisbelow(True)
    figure.tight_layout()

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(OUTPUT_PATH, dpi=200)
    plt.close(figure)
    print(f"saved {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
