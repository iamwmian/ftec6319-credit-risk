"""Slide figures for the part 15 strategy thresholds.

Three views of the same train-sample search:

- default rate against the threshold
- 12-month revenue against the threshold
- revenue against default rate, which is the management tradeoff

The solid part of each curve meets the firm's 10% default-rate limit.
The dashed part breaks that limit. The two marked points are the
conservative threshold and the aggressive threshold.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parent.parent
STRATEGY_DIR = PROJECT_DIR / "strategy"
SEARCH_PATH = STRATEGY_DIR / "threshold_search.csv"
CHOICE_PATH = STRATEGY_DIR / "chosen_strategies.csv"

CURVE = "#4B5563"
OVER_LIMIT = "#B0B0B0"
CONSERVATIVE = "#1D4E89"
AGGRESSIVE = "#C2410C"
LIMIT = "#222222"

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


def load():
    search = pd.read_csv(SEARCH_PATH).sort_values("threshold")
    chosen = pd.read_csv(CHOICE_PATH)
    return search, chosen


def split_limit(search):
    """Return the curve inside the 10% cap and the curve outside it."""
    allowed = search.loc[search["within_firm_limit"]]
    blocked = search.loc[~search["within_firm_limit"]]
    # Keep one shared point so the solid and dashed pieces meet.
    if len(allowed) and len(blocked):
        blocked = pd.concat([allowed.tail(1), blocked], ignore_index=True)
    return allowed, blocked


def draw_curve(axis, allowed, blocked, x_column, y_column):
    axis.plot(
        allowed[x_column],
        allowed[y_column],
        color=CURVE,
        linewidth=2.2,
        label="Within 10% limit",
        zorder=2,
    )
    axis.plot(
        blocked[x_column],
        blocked[y_column],
        color=OVER_LIMIT,
        linewidth=2.2,
        linestyle="--",
        label="Above 10% limit",
        zorder=2,
    )


def mark_choices(axis, chosen, x_column, y_column):
    colors = {"conservative": CONSERVATIVE, "aggressive": AGGRESSIVE}
    labels = {"conservative": "Conservative", "aggressive": "Aggressive"}
    for row in chosen.itertuples(index=False):
        axis.scatter(
            [getattr(row, x_column)],
            [getattr(row, y_column)],
            s=160,
            color=colors[row.strategy],
            zorder=3,
            label=f"{labels[row.strategy]}, t = {row.threshold:.2f}",
        )


def finish(axis, title, xlabel, ylabel, path):
    axis.set_title(title)
    axis.set_xlabel(xlabel)
    axis.set_ylabel(ylabel)
    axis.grid(True, color="#E6E6E6", linewidth=0.8)
    axis.set_axisbelow(True)
    axis.legend(frameon=False, loc="upper left")
    path.parent.mkdir(parents=True, exist_ok=True)
    axis.figure.tight_layout()
    axis.figure.savefig(path, dpi=200)
    plt.close(axis.figure)
    print(f"saved {path}")


def plot_default_rate(search, chosen):
    figure, axis = plt.subplots(figsize=(11, 7.2))
    allowed, blocked = split_limit(search)
    draw_curve(
        axis,
        allowed.assign(default_pct=allowed["default_rate"] * 100),
        blocked.assign(default_pct=blocked["default_rate"] * 100),
        "threshold",
        "default_pct",
    )
    axis.axhline(5, color=LIMIT, linestyle=":", linewidth=1.3, label="5% conservative cap")
    axis.axhline(10, color=LIMIT, linestyle="--", linewidth=1.3, label="10% firm limit")
    marked = chosen.assign(default_pct=chosen["default_rate"] * 100)
    mark_choices(axis, marked, "threshold", "default_pct")
    axis.set_xlim(0, 1)
    axis.set_ylim(0, 30)
    axis.yaxis.set_major_formatter(plt.FuncFormatter(lambda value, _position: f"{value:.0f}%"))
    axis.xaxis.set_major_formatter(plt.FuncFormatter(lambda value, _position: f"{value:.1f}"))
    finish(
        axis,
        "Default rate of accepted accounts",
        "Threshold",
        "Portfolio default rate",
        STRATEGY_DIR / "default_rate_by_threshold.png",
    )


def plot_revenue(search, chosen):
    figure, axis = plt.subplots(figsize=(11, 7.2))
    allowed, blocked = split_limit(search)
    draw_curve(axis, allowed, blocked, "threshold", "expected_revenue")
    mark_choices(axis, chosen, "threshold", "expected_revenue")
    axis.set_xlim(0, 1)
    axis.yaxis.set_major_formatter(plt.FuncFormatter(lambda value, _position: f"{value:,.0f}"))
    axis.xaxis.set_major_formatter(plt.FuncFormatter(lambda value, _position: f"{value:.1f}"))
    finish(
        axis,
        "12-month revenue of accepted accounts",
        "Threshold",
        "Portfolio revenue",
        STRATEGY_DIR / "revenue_by_threshold.png",
    )


def plot_tradeoff(search, chosen):
    figure, axis = plt.subplots(figsize=(11, 7.2))
    rated = search.assign(default_pct=search["default_rate"] * 100)
    allowed, blocked = split_limit(rated)
    draw_curve(axis, allowed, blocked, "default_pct", "expected_revenue")
    axis.axvline(5, color=LIMIT, linestyle=":", linewidth=1.3, label="5% conservative cap")
    axis.axvline(10, color=LIMIT, linestyle="--", linewidth=1.3, label="10% firm limit")
    marked = chosen.assign(default_pct=chosen["default_rate"] * 100)
    mark_choices(axis, marked, "default_pct", "expected_revenue")
    axis.set_xlim(0, 30)
    axis.xaxis.set_major_formatter(plt.FuncFormatter(lambda value, _position: f"{value:.0f}%"))
    axis.yaxis.set_major_formatter(plt.FuncFormatter(lambda value, _position: f"{value:,.0f}"))
    finish(
        axis,
        "Revenue against default rate",
        "Portfolio default rate",
        "Portfolio revenue",
        STRATEGY_DIR / "revenue_vs_default_rate.png",
    )


def main():
    search, chosen = load()
    plot_default_rate(search, chosen)
    plot_revenue(search, chosen)
    plot_tradeoff(search, chosen)


if __name__ == "__main__":
    main()
