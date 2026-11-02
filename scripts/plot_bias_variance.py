"""Slide figures for the XGBoost and neural-net model choices.

Each grid setting is one point. The horizontal axis is the average of the
train, Test 1, and Test 2 AUCs. Farther right means less bias. The vertical
axis is the spread of those three AUCs. Higher means more variance.

The selection score is average minus spread. The dashed line is the winning
score. The chosen model sits on that line, and every other model is above it.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parent.parent
FIGURES = [
    (
        PROJECT_DIR / "results" / "05_xgb_grid_bias_variance.csv",
        PROJECT_DIR / "results" / "08_xgb_bias_variance.png",
    ),
    (
        PROJECT_DIR / "results" / "11_nn_grid_bias_variance.csv",
        PROJECT_DIR / "results" / "15_nn_bias_variance.png",
    ),
]

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


def axis_limits(grid):
    """Limits that keep the winning-score line a visible diagonal.

    That line has slope 1, because spread = average - score. The XGBoost
    averages and spreads cover about the same width, so the full grid is
    already readable. The neural-net grid is not: a few weak settings sit
    far to the left, and the line then becomes a vertical mark. The
    horizontal window stays at most 1.2 times the spread window, anchored
    at the best average. A grid that is already inside that width is unchanged.
    """
    x_pad = 0.001
    y_pad = 0.002
    x_right = float(grid["mean_auc"].max()) + x_pad
    x_left = float(grid["mean_auc"].min()) - x_pad
    for _ in range(5):
        visible = grid.loc[grid["mean_auc"] >= x_left]
        y_top = float(visible["std_auc"].max()) + y_pad
        x_left = max(float(grid["mean_auc"].min()) - x_pad, x_right - 1.2 * y_top)
    return x_left, x_right, y_top


def plot_figure(grid_path, output_path):
    grid = pd.read_csv(grid_path)
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
    x_left, x_right, y_top = axis_limits(grid)
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
    axis.set_ylim(0, y_top)
    axis.xaxis.set_major_formatter(plt.FuncFormatter(lambda value, _position: f"{value:.3f}"))
    axis.yaxis.set_major_formatter(plt.FuncFormatter(lambda value, _position: f"{value:.3f}"))
    axis.legend(frameon=False, loc="upper left")
    axis.grid(True, color="#E6E6E6", linewidth=0.8)
    axis.set_axisbelow(True)
    figure.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=200)
    plt.close(figure)
    print(f"saved {output_path}")


def main():
    for grid_path, output_path in FIGURES:
        plot_figure(grid_path, output_path)


if __name__ == "__main__":
    main()
